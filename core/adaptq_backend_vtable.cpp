#include "../include/adaptq_backend.h"
#include "../include/adaptq/backend_selection.h"
#include <cstdlib>
#include <cstring>

// Shims that cast impl to IAdaptQBackend* and forward calls
static void s_append_kv(void *c, int h, const float *k, const float *v, int p) {
  static_cast<IAdaptQBackend *>(c)->append_kv(h, k, v, p);
}
static int s_compute(void *c, int h, const float *q, float *o) {
  return static_cast<IAdaptQBackend *>(c)->compute(h, q, o);
}
static int s_compute_batch(void *c, int h, const float *qs, int n, float *os) {
  return static_cast<IAdaptQBackend *>(c)->compute_batch(h, qs, n, os);
}
static void s_reset(void *c) { static_cast<IAdaptQBackend *>(c)->reset(); }
static size_t s_kv_bytes(void *c) {
  return static_cast<IAdaptQBackend *>(c)->kv_bytes();
}
static int s_n_heads(void *c) {
  return static_cast<IAdaptQBackend *>(c)->n_heads();
}
static int s_head_dim(void *c) {
  return static_cast<IAdaptQBackend *>(c)->head_dim();
}
static void s_destroy(void *c) { delete static_cast<IAdaptQBackend *>(c); }

AdaptQBackendVTable adaptq_make_vtable(IAdaptQBackend *backend) {
  return {s_append_kv, s_compute,  s_compute_batch, s_reset, s_kv_bytes,
          s_n_heads,   s_head_dim, s_destroy,       backend};
}

namespace adaptq {

bool is_scalar_forced() {
  const char *disable_avx2 = std::getenv("ADAPTQ_DISABLE_AVX2");
  if (disable_avx2 && (std::strcmp(disable_avx2, "1") == 0 ||
                       std::strcmp(disable_avx2, "true") == 0 ||
                       std::strcmp(disable_avx2, "TRUE") == 0)) {
    return true;
  }

  const char *force_scalar = std::getenv("ADAPTQ_FORCE_SCALAR");
  if (force_scalar && (std::strcmp(force_scalar, "1") == 0 ||
                       std::strcmp(force_scalar, "true") == 0 ||
                       std::strcmp(force_scalar, "TRUE") == 0)) {
    return true;
  }

  const char *backend = std::getenv("ADAPTQ_BACKEND");
  return backend && (std::strcmp(backend, "scalar") == 0 ||
                     std::strcmp(backend, "SCALAR") == 0);
}

} /* namespace adaptq */
