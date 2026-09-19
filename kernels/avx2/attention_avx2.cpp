#include "../../include/adaptq/attention_avx2.h"
#include "softmax_avx2.h"
#include "../../include/codebook.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#define ADAPTQ_HAS_AVX2 1
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__)
#define ADAPTQ_HAS_AVX2 1
#include <intrin.h>
#include <immintrin.h>
#else
#define ADAPTQ_HAS_AVX2 0
#endif

#if ADAPTQ_HAS_AVX2


// Permutevar 4-bit lookup: 8 indices in ~8 cycles vs ~40 for gather
static inline __m256 lup8(const __m256i idx, const __m256 cl, const __m256 ch) {
  const __m256i m7 = _mm256_set1_epi32(7);
  __m256i gt7 = _mm256_cmpgt_epi32(idx, m7);
  __m256i i7 = _mm256_and_si256(idx, m7);
  return _mm256_blendv_ps(_mm256_permutevar8x32_ps(cl, i7),
                          _mm256_permutevar8x32_ps(ch, i7),
                          _mm256_castsi256_ps(gt7));
}

// ---------------------------------------------------------------------------
// Templatized Decoding logic: extracts 8 indices into an int32×8 vector
// for permutevar8x32.
// ---------------------------------------------------------------------------
template <int BITS> struct Decode;

template <> struct Decode<4> {
  static constexpr int step = 4;
  static inline __m256i f(const uint8_t *p) {
    uint32_t val;
    memcpy(&val, p, 4);
    __m128i h =
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)((val >> 4) & 0x0F0F0F0F)));
    __m128i l = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)(val & 0x0F0F0F0F)));
    return _mm256_set_m128i(_mm_unpackhi_epi32(h, l), _mm_unpacklo_epi32(h, l));
  }
};

template <> struct Decode<3> {
  static constexpr int step = 3;
  static inline __m256i f(const uint8_t *p) {
    uint32_t val =
        ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
    __m256i v256 = _mm256_set1_epi32(val);
    __m256i shifts = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
    return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts),
                            _mm256_set1_epi32(7));
  }
};

template <> struct Decode<2> {
  static constexpr int step = 2;
  static inline __m256i f(const uint8_t *p) {
    uint16_t val;
    memcpy(&val, p, 2);
    __m256i v256 = _mm256_set1_epi32(val);
    __m256i shifts = _mm256_setr_epi32(6, 4, 2, 0, 14, 12, 10, 8);
    return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts),
                            _mm256_set1_epi32(3));
  }
};

// Horizontal sum of __m256 → scalar (uses XMM path, no extra YMM pressure)
static inline float hsum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}


// ---------------------------------------------------------------------------
// SINGLE-TOKEN K-dot: used for tail processing
// ---------------------------------------------------------------------------
template <int BITS>
static float kdot1(const float *__restrict qr, const uint8_t *__restrict pk,
                   const __m256 cl, const __m256 ch, int padded) {
  __m256 s0 = _mm256_setzero_ps();
  int b = 0;
  for (int j = 0; j < padded; j += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + j),
                         lup8(Decode<BITS>::f(pk + b), cl, ch), s0);
    b += Decode<BITS>::step;
  }
  return hsum8(s0);
}

// ---------------------------------------------------------------------------
// 4-TOKEN K-dot: all 4 tokens processed in one loop, sharing q_rot loads.
// ---------------------------------------------------------------------------
template <int BITS>
static void kdot4_quad(const float *__restrict qr, const uint8_t *__restrict k0,
                       const uint8_t *__restrict k1,
                       const uint8_t *__restrict k2,
                       const uint8_t *__restrict k3, const __m256 cl,
                       const __m256 ch, int padded, float out[4]) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
  int b = 0;
  for (int j = 0; j < padded; j += 16) {
    __m256 qL = _mm256_loadu_ps(qr + j);
    s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0 + b), cl, ch), qL, s0);
    s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1 + b), cl, ch), qL, s1);
    s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2 + b), cl, ch), qL, s2);
    s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3 + b), cl, ch), qL, s3);
    b += Decode<BITS>::step;
    __m256 qH = _mm256_loadu_ps(qr + j + 8);
    s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0 + b), cl, ch), qH, s0);
    s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1 + b), cl, ch), qH, s1);
    s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2 + b), cl, ch), qH, s2);
    s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3 + b), cl, ch), qH, s3);
    b += Decode<BITS>::step;
  }
  out[0] = hsum8(s0);
  out[1] = hsum8(s1);
  out[2] = hsum8(s2);
  out[3] = hsum8(s3);
}

// ---------------------------------------------------------------------------
// V-accum 4-token batch: 1 load + 4 FMAs + 1 store per 16-pos block.
// ---------------------------------------------------------------------------
template <int BITS>
static void vaccum4(float *__restrict acc, const uint8_t *v0, const uint8_t *v1,
                    const uint8_t *v2, const uint8_t *v3, float e0, float e1,
                    float e2, float e3, const __m256 cl, const __m256 ch,
                    int padded) {
  __m256 q0 = _mm256_set1_ps(e0), q1 = _mm256_set1_ps(e1);
  __m256 q2 = _mm256_set1_ps(e2), q3 = _mm256_set1_ps(e3);
  int b = 0;
  for (int j = 0; j < padded; j += 16) {
    __m256 ra = _mm256_loadu_ps(acc + j);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0 + b), cl, ch), q0, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1 + b), cl, ch), q1, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2 + b), cl, ch), q2, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3 + b), cl, ch), q3, ra);
    _mm256_storeu_ps(acc + j, ra);
    b += Decode<BITS>::step;

    __m256 rb = _mm256_loadu_ps(acc + j + 8);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0 + b), cl, ch), q0, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1 + b), cl, ch), q1, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2 + b), cl, ch), q2, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3 + b), cl, ch), q3, rb);
    _mm256_storeu_ps(acc + j + 8, rb);
    b += Decode<BITS>::step;
  }
}

template <int BITS>
static void vaccum1(float *__restrict acc, const uint8_t *__restrict vp,
                    float ew, const __m256 cl, const __m256 ch, int padded) {
  __m256 ev = _mm256_set1_ps(ew);
  int b = 0;
  for (int j = 0; j < padded; j += 8) {
    __m256 ra = _mm256_loadu_ps(acc + j);
    _mm256_storeu_ps(
        acc + j,
        _mm256_fmadd_ps(lup8(Decode<BITS>::f(vp + b), cl, ch), ev, ra));
    b += Decode<BITS>::step;
  }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC pop_options
#endif



/* Fast AVX2 exp approximation adapted from the standard minimax/cephes form. */
static inline __m256 exp8_avx2(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
  const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 c1 = _mm256_set1_ps(6.93359375e-1f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);

  x = _mm256_max_ps(exp_lo, _mm256_min_ps(exp_hi, x));

  __m256 fx = _mm256_fmadd_ps(x, log2ef, half);
  fx = _mm256_floor_ps(fx);

  __m256 tmp = _mm256_mul_ps(fx, c1);
  x = _mm256_sub_ps(x, tmp);
  tmp = _mm256_mul_ps(fx, c2);
  x = _mm256_sub_ps(x, tmp);

  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);

  __m256i emm0 = _mm256_cvttps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
  emm0 = _mm256_slli_epi32(emm0, 23);
  const __m256 pow2n = _mm256_castsi256_ps(emm0);
  return _mm256_mul_ps(y, pow2n);
}

void softmax_avx2(float *x, int n) {
  if (!x || n <= 0)
    return;
  if (n == 1) {
    x[0] = 1.0f;
    return;
  }

  float mx = x[0];
  for (int i = 1; i < n; ++i)
    if (x[i] > mx)
      mx = x[i];

  const __m256 mxv = _mm256_set1_ps(mx);
  __m256 sumv = _mm256_setzero_ps();
  int i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    v = exp8_avx2(_mm256_sub_ps(v, mxv));
    _mm256_storeu_ps(x + i, v);
    sumv = _mm256_add_ps(sumv, v);
  }

  float sum = hsum8(sumv);
  for (; i < n; ++i) {
    x[i] = expf(x[i] - mx);
    sum += x[i];
  }

  if (sum > 0.f && std::isfinite(sum)) {
    const float inv = 1.f / sum;
    for (i = 0; i < n; ++i)
      x[i] *= inv;
  } else {
    const float uniform = 1.f / (float)n;
    for (i = 0; i < n; ++i)
      x[i] = uniform;
  }
}


template <int BITS>
static void compute_avx2(const float *qr, float *acc, const float *cb,
                         const uint8_t *kb, const uint8_t *vb,
                         const float *kscale, const float *vscale, float attn_s,
                         float isp, int *slots, int n, int pb, int padded,
                         float v_mass_thresh, int *ord, float *logits) {
  const __m256 cl = _mm256_loadu_ps(cb), ch = _mm256_loadu_ps(cb + 8);
  int i = 0;
  for (; i + 3 < n; i += 4) {
    int s0 = slots[i], s1 = slots[i + 1], s2 = slots[i + 2], s3 = slots[i + 3];
    if (i + 7 < n) {
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 4] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 5] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 6] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 7] * pb);
    }
    ADAPTQ_PREFETCH(vb + (size_t)s0 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s1 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s2 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s3 * pb);

    float d[4];
    kdot4_quad<BITS>(qr, kb + (size_t)s0 * pb, kb + (size_t)s1 * pb,
                     kb + (size_t)s2 * pb, kb + (size_t)s3 * pb, cl, ch, padded,
                     d);
    logits[i] = d[0] * attn_s * kscale[s0];
    logits[i + 1] = d[1] * attn_s * kscale[s1];
    logits[i + 2] = d[2] * attn_s * kscale[s2];
    logits[i + 3] = d[3] * attn_s * kscale[s3];
  }
  for (; i < n; ++i) {
    int s = slots[i];
    ADAPTQ_PREFETCH(vb + (size_t)s * pb);
    logits[i] = kdot1<BITS>(qr, kb + (size_t)s * pb, cl, ch, padded) * attn_s *
                kscale[s];
  }

  softmax_avx2(logits, n);

  memset(acc, 0, padded * sizeof(float));

  if (v_mass_thresh <= 0.f) {
    int ii = 0;
    for (; ii + 3 < n; ii += 4) {
      int s0 = slots[ii], s1 = slots[ii + 1], s2 = slots[ii + 2],
          s3 = slots[ii + 3];
      vaccum4<BITS>(
          acc, vb + (size_t)s0 * pb, vb + (size_t)s1 * pb, vb + (size_t)s2 * pb,
          vb + (size_t)s3 * pb, logits[ii] * vscale[s0] * isp,
          logits[ii + 1] * vscale[s1] * isp, logits[ii + 2] * vscale[s2] * isp,
          logits[ii + 3] * vscale[s3] * isp, cl, ch, padded);
    }
    for (; ii < n; ++ii) {
      int s = slots[ii];
      vaccum1<BITS>(acc, vb + (size_t)s * pb, logits[ii] * vscale[s] * isp, cl,
                    ch, padded);
    }
  } else {
    for (int ii = 0; ii < n; ++ii)
      ord[ii] = ii;
    std::sort(ord, ord + n,
              [&](int a, int b) { return logits[a] > logits[b]; });
    float mass = 0.f;
    int ii = 0;
    for (; ii + 3 < n && mass < v_mass_thresh; ii += 4) {
      int i0 = ord[ii], i1 = ord[ii + 1], i2 = ord[ii + 2], i3 = ord[ii + 3];
      int s0 = slots[i0], s1 = slots[i1], s2 = slots[i2], s3 = slots[i3];
      vaccum4<BITS>(
          acc, vb + (size_t)s0 * pb, vb + (size_t)s1 * pb, vb + (size_t)s2 * pb,
          vb + (size_t)s3 * pb, logits[i0] * vscale[s0] * isp,
          logits[i1] * vscale[s1] * isp, logits[i2] * vscale[s2] * isp,
          logits[i3] * vscale[s3] * isp, cl, ch, padded);
      mass += logits[i0] + logits[i1] + logits[i2] + logits[i3];
    }
    for (; ii < n && mass < v_mass_thresh; ++ii) {
      int s = slots[ord[ii]];
      float w = logits[ord[ii]];
      vaccum1<BITS>(acc, vb + (size_t)s * pb, w * vscale[s] * isp, cl, ch,
                    padded);
      mass += w;
    }
  }
}


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC pop_options
#endif

#if defined(_MSC_VER)
static bool host_supports_avx2_fma() {
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 1)
        return false;

    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma)
        return false;

    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL)
        return false;

    __cpuidex(regs, 0, 0);
    if (regs[0] < 7)
        return false;

    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
}
#elif defined(__GNUC__) || defined(__clang__)
static bool host_supports_avx2_fma() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
static bool host_supports_avx2_fma() {
    return false;
}
#endif

#endif // ADAPTQ_HAS_AVX2

bool adaptq_attention_avx2_compute(
    const float *q_rot, float *acc, const float *codebook,
    const uint8_t *k_data, const uint8_t *v_data,
    const float *k_scale, const float *v_scale,
    float attention_scale, float inverse_sqrt_padded,
    int *slots, int token_count, int packed_bytes, int padded, int bits,
    float value_mass_threshold, int *order, float *logits) {
#if ADAPTQ_HAS_AVX2
    if (!host_supports_avx2_fma())
        return false;

    switch (bits) {
    case 4:
        compute_avx2<4>(q_rot, acc, codebook, k_data, v_data, k_scale, v_scale,
                        attention_scale, inverse_sqrt_padded, slots,
                        token_count, packed_bytes, padded, value_mass_threshold,
                        order, logits);
        return true;
    case 3:
        compute_avx2<3>(q_rot, acc, codebook, k_data, v_data, k_scale, v_scale,
                        attention_scale, inverse_sqrt_padded, slots,
                        token_count, packed_bytes, padded, value_mass_threshold,
                        order, logits);
        return true;
    case 2:
        compute_avx2<2>(q_rot, acc, codebook, k_data, v_data, k_scale, v_scale,
                        attention_scale, inverse_sqrt_padded, slots,
                        token_count, packed_bytes, padded, value_mass_threshold,
                        order, logits);
        return true;
    default:
        return false;
    }
#else
    (void)q_rot; (void)acc; (void)codebook; (void)k_data; (void)v_data;
    (void)k_scale; (void)v_scale; (void)attention_scale;
    (void)inverse_sqrt_padded; (void)slots; (void)token_count;
    (void)packed_bytes; (void)padded; (void)bits;
    (void)value_mass_threshold; (void)order; (void)logits;
    return false;
#endif
}
