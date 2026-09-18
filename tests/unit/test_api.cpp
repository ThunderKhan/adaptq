/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <string>

/* -------------------------------------------------------------------------
 * tests/unit/test_api.cpp
 * C ABI smoke tests — covers every public function in adaptq.h.
 * ----------------------------------------------------------------------- */

static void fill_vec(float *v, int n, float val) {
    for (int i = 0; i < n; ++i) v[i] = val;
}

/* ---- Single head -------------------------------------------------------- */

TEST_CASE("adaptq_create returns non-null handle", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 42, 0.f, 0);