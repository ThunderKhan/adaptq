#include "../include/attention.h"
#include "../include/adaptq/attention_avx2.h"
#include "../include/codebook.h"
#include <algorithm>
#include <cmath>
#include <cstring>

/* Portable cache prefetch — __builtin_prefetch is GCC/Clang only */
#if defined(_MSC_VER)
#  include <xmmintrin.h>
#  define ADAPTQ_PREFETCH(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1)
#else
#  define ADAPTQ_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 1)
#endif

#include <vector>

struct AttentionWorkspace {
  std::vector<float> logits;
  std::vector<int> slots;
  std::vector<int> ord;
  std::vector<float> q_rot;
  std::vector<float> v_accum;

  void ensure_capacity(int n, int padded) {
    if (logits.size() < (size_t)n) {
      logits.resize(n);
      slots.resize(n);
      ord.resize(n);
    }
    if (q_rot.size() < (size_t)padded) {
      q_rot.resize(padded);
      v_accum.resize(padded);
    }
  }
};

static thread_local AttentionWorkspace tl_ws;

float dot_product(const float *a, const float *b, int n) {
  if (!a || !b || n <= 0)
    return 0.f;
  float s = 0.f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}
void softmax(float *x, int n) {
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
  float s = 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] = expf(x[i] - mx);
    s += x[i];
  }
  if (s > 0.f && std::isfinite(s)) {
    float inv = 1.f / s;
    for (int i = 0; i < n; ++i)
      x[i] *= inv;
  } else {
    float unif = 1.f / (float)n;
    for (int i = 0; i < n; ++i)
      x[i] = unif;
  }
}
[[maybe_unused]]
static float kdot_scalar(const float *q, const uint8_t *p, const float *cb,
                         int nb, int bits) {
  float s = 0.f;
  if (bits == 2) {
    for (int i = 0; i < nb; ++i) {
      uint8_t v = p[i];
      int j = i << 2;
      s += q[j] * cb[(v >> 6) & 3] + q[j + 1] * cb[(v >> 4) & 3] +
           q[j + 2] * cb[(v >> 2) & 3] + q[j + 3] * cb[v & 3];
    }
  } else if (bits == 3) {
    int g = nb / 3;
    for (int i = 0; i < g; ++i) {
      const uint8_t *r = p + i * 3;
      int j = i * 8;
      s += q[j] * cb[(r[0] >> 5) & 7] + q[j + 1] * cb[(r[0] >> 2) & 7] +
           q[j + 2] * cb[((r[0] & 3) << 1) | (r[1] >> 7)] +
           q[j + 3] * cb[(r[1] >> 4) & 7] + q[j + 4] * cb[(r[1] >> 1) & 7] +
           q[j + 5] * cb[((r[1] & 1) << 2) | (r[2] >> 6)] +
           q[j + 6] * cb[(r[2] >> 3) & 7] + q[j + 7] * cb[r[2] & 7];
    }
  } else {
    for (int i = 0; i + 3 < nb; i += 4) {
      uint8_t b0 = p[i], b1 = p[i + 1], b2 = p[i + 2], b3 = p[i + 3];
      int j = i << 1;
      s += q[j] * cb[b0 >> 4] + q[j + 1] * cb[b0 & 15] +
           q[j + 2] * cb[b1 >> 4] + q[j + 3] * cb[b1 & 15] +
           q[j + 4] * cb[b2 >> 4] + q[j + 5] * cb[b2 & 15] +
           q[j + 6] * cb[b3 >> 4] + q[j + 7] * cb[b3 & 15];
    }
  }
  return s;
}

static void vaccum_scalar(float *acc, const uint8_t *vp, const float *ecb,
                          int nb, int bits) {
  if (bits == 2) {
    for (int b = 0; b < nb; ++b) {
      uint8_t v = vp[b];
      int j = b << 2;
      acc[j] += ecb[(v >> 6) & 3];
      acc[j + 1] += ecb[(v >> 4) & 3];
      acc[j + 2] += ecb[(v >> 2) & 3];
      acc[j + 3] += ecb[v & 3];
    }
  } else if (bits == 3) {
    int g = nb / 3;
    for (int i = 0; i < g; ++i) {
      const uint8_t *r = vp + i * 3;
      int j = i * 8;
      acc[j] += ecb[(r[0] >> 5) & 7];
      acc[j + 1] += ecb[(r[0] >> 2) & 7];
      acc[j + 2] += ecb[((r[0] & 3) << 1) | (r[1] >> 7)];
      acc[j + 3] += ecb[(r[1] >> 4) & 7];
      acc[j + 4] += ecb[(r[1] >> 1) & 7];
      acc[j + 5] += ecb[((r[1] & 1) << 2) | (r[2] >> 6)];
      acc[j + 6] += ecb[(r[2] >> 3) & 7];
      acc[j + 7] += ecb[r[2] & 7];
    }
  } else {
    for (int b = 0; b + 3 < nb; b += 4) {
      uint8_t b0 = vp[b], b1 = vp[b + 1], b2 = vp[b + 2], b3 = vp[b + 3];
      int j = b << 1;
      acc[j] += ecb[b0 >> 4];
      acc[j + 1] += ecb[b0 & 15];
      acc[j + 2] += ecb[b1 >> 4];
      acc[j + 3] += ecb[b1 & 15];
      acc[j + 4] += ecb[b2 >> 4];
      acc[j + 5] += ecb[b2 & 15];
      acc[j + 6] += ecb[b3 >> 4];
      acc[j + 7] += ecb[b3 & 15];
    }
  }
}

// ---------------------------------------------------------------------------
void AttentionHead::init(int d, int b, int cap, uint64_t seed, float v_mass,
                         int hyb) {
  dim = d;
  bits = b;
  v_mass_thresh = v_mass;
  hybrid_thresh = hyb;
  quant.init(d, seed);
  padded = quant.padded;
  kv_buf.init(cap, padded, b);
  raw_kv.clear();
  if (hyb > 0)
    raw_kv.reserve((size_t)hyb * 2 * d);
}
void AttentionHead::append_kv(const float *key, const float *val, int pos) {
  static thread_local uint8_t tmp_k[8192], tmp_v[8192];
  float ks = quant.quantize_into(key, bits, tmp_k);
  float vs = quant.quantize_into(val, bits, tmp_v);
  bool will_evict = (kv_buf.size >= kv_buf.capacity);
  kv_buf.insert(tmp_k, ks, tmp_v, vs, pos);
  // Mirror raw floats for hybrid FP path (only up to threshold, before circular eviction)
  if (hybrid_thresh > 0) {
    if (!will_evict && (int)raw_kv.size() < hybrid_thresh * 2 * dim) {
      raw_kv.insert(raw_kv.end(), key, key + dim);
      raw_kv.insert(raw_kv.end(), val, val + dim);
    } else if (will_evict && !raw_kv.empty()) {
      // Invalidate raw_kv once circular FIFO eviction occurs to prevent serving stale tokens
      raw_kv.clear();
    }
  }
}

int AttentionHead::compute(const float *q, float *out) const {
  const int n = kv_buf.size, cap = kv_buf.capacity, pb = kv_buf.packed_bytes;
  if (!n) {
    memset(out, 0, dim * sizeof(float));
    return 0;
  }

  // ---- Hybrid path: FP32 attention for small sequences -------------------
  // Below hybrid_thresh, FP32 is faster (data fits in L1/L2, no decode cost).
  // raw_kv stores interleaved [k0...kd, v0...vd, k1...] for the first
  // hybrid_thresh tokens. Zero-overhead check: one integer compare.
  if (hybrid_thresh > 0 && n <= hybrid_thresh &&
      (int)raw_kv.size() == n * 2 * dim) {
    tl_ws.ensure_capacity(n, padded);
    float *logits = tl_ws.logits.data();
    const float scale = 1.f / sqrtf((float)dim);
    float mx = -1e30f;
    // Compute all logits in one pass, track max
    for (int i = 0; i < n; ++i) {
      const float *k = raw_kv.data() + (size_t)i * 2 * dim;
      float d = 0.f;
      for (int j = 0; j < dim; ++j)
        d += q[j] * k[j];
      logits[i] = d * scale;
      if (logits[i] > mx)
        mx = logits[i];
    }
    // Softmax: 2 passes (max known)
    float sv = 0.f;
    for (int i = 0; i < n; ++i) {
      logits[i] = expf(logits[i] - mx);
      sv += logits[i];
    }
    float inv = (sv > 0.f && std::isfinite(sv)) ? (1.f / sv) : (1.f / (float)n);
    // Weighted V accumulation
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n; ++i) {
      float w = logits[i] * inv;
      const float *v = raw_kv.data() + (size_t)i * 2 * dim + dim;
      for (int j = 0; j < dim; ++j)
        out[j] += w * v[j];
    }
    return n;
  }
  // ---- End hybrid path ---------------------------------------------------

  tl_ws.ensure_capacity(n, padded);

  // Rotate query
  float *qr = tl_ws.q_rot.data();
  memcpy(qr, q, dim * sizeof(float));
  for (int i = dim; i < padded; ++i)
    qr[i] = 0.f;
  float qn = 0.f;
  for (int i = 0; i < dim; ++i)
    qn += q[i] * q[i];
  qn = sqrtf(qn + 1e-12f);
  {
    float inv = 1.f / qn;
    for (int i = 0; i < padded; ++i)
      qr[i] *= inv;
  }
  fwht_forward(qr, quant.D.data(), padded);
  float sp = sqrtf((float)padded);
  for (int i = 0; i < padded; ++i)
    qr[i] *= sp * qn;

  const float *cb = get_codebook(bits);
  float *acc = tl_ws.v_accum.data();
  const uint8_t *kb = kv_buf.k_data;
  const uint8_t *vb = kv_buf.v_data;
  float attn_s = 1.f / (sqrtf((float)dim) * (float)padded);
  float isp = 1.f / sqrtf((float)padded);
  float *logits = tl_ws.logits.data();
  int *slots = tl_ws.slots.data();
  for (int i = 0; i < n; ++i)
    slots[i] = (kv_buf.head - n + cap + i) % cap;



  // inside compute():
  if (adaptq_attention_avx2_compute(
          qr, acc, cb, kb, vb, kv_buf.k_scale.data(),
          kv_buf.v_scale.data(), attn_s, isp, tl_ws.slots.data(), n, pb, padded,
          bits, v_mass_thresh, tl_ws.ord.data(), logits)) {
    fwht_inverse(acc, quant.D.data(), padded);
    memcpy(out, acc, dim * sizeof(float));
    return n;
  }

  // Scalar 2/3-bit path
  memset(acc, 0, padded * sizeof(float));
  for (int i = 0; i < n; ++i) {
    int s = slots[i];
    if (i + 4 < n) {
      int ps = slots[i + 4];
      ADAPTQ_PREFETCH(kb + (size_t)ps * pb);
      ADAPTQ_PREFETCH(vb + (size_t)ps * pb);
    }
    logits[i] = kdot_scalar(qr, kb + (size_t)s * pb, cb, pb, bits) * attn_s *
                kv_buf.k_scale[s];
  }
  softmax(logits, n);
  int cb_sz = 1 << bits;
  for (int i = 0; i < n; ++i) {
    int s = slots[i];
    float ecb[16];
    float ew = logits[i] * kv_buf.v_scale[s] * isp;
    for (int k = 0; k < cb_sz; ++k)
      ecb[k] = ew * cb[k];
    vaccum_scalar(acc, vb + (size_t)s * pb, ecb, pb, bits);
  }
  fwht_inverse(acc, quant.D.data(), padded);
  memcpy(out, acc, dim * sizeof(float));
  return n;
}

int AttentionHead::compute_batch(const float *queries, int num_queries,
                                 float *outs) const {
  if (kv_buf.size == 0) {
    memset(outs, 0, num_queries * dim * sizeof(float));
    return 0;
  }

  // Auto-tune batch threading depending on active queries
  // thread_local buffers in compute() ensure OpenMP safety
#pragma omp parallel for if (num_queries > 1)
  for (int q = 0; q < num_queries; ++q) {
    compute(queries + q * dim, outs + q * dim);
  }

  return kv_buf.size;
}
