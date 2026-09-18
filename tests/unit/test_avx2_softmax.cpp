#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
static void set_test_env(const char *name, const char *value) {
    _putenv_s(name, value);
}
static void unset_test_env(const char *name) {
    _putenv_s(name, "");
}
#else
static void set_test_env(const char *name, const char *value) {
    setenv(name, value, 1);
}
static void unset_test_env(const char *name) {
    unsetenv(name);
}
#endif

TEST_CASE("AVX2 softmax matches scalar attention path", "[attention][avx2]") {
    adaptq_ctx_t h = adaptq_create(64, 4, 32, 123, 0.f, 0);
    REQUIRE(h != nullptr);

    std::vector<float> k(64), v(64), q(64), scalar_out(64), avx2_out(64);
    for (int i = 0; i < 64; ++i) {
        k[i] = std::sin((float)i * 0.17f) * 0.8f + 0.1f;
        v[i] = std::cos((float)i * 0.11f) * 0.6f - 0.2f;
        q[i] = std::sin((float)i * 0.07f + 0.3f);
    }
    for (int token = 0; token < 16; ++token) {
        adaptq_append(h, k.data(), v.data(), token);
        std::rotate(k.begin(), k.begin() + 1, k.end());
        std::rotate(v.begin(), v.begin() + 3, v.end());
    }

    set_test_env("ADAPTQ_DISABLE_AVX2", "1");
    REQUIRE(adaptq_compute(h, q.data(), scalar_out.data()) == 16);
    unset_test_env("ADAPTQ_DISABLE_AVX2");

    REQUIRE(adaptq_compute(h, q.data(), avx2_out.data()) == 16);

    for (int i = 0; i < 64; ++i)
        REQUIRE(std::abs(scalar_out[i] - avx2_out[i]) < 2e-3f);

    adaptq_destroy(h);
}
