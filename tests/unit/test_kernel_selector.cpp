/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq/kernel.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------
 * tests/unit/test_kernel_selector.cpp
 *
 * Issue #7: Automated AVX2 capability detection, XCR0 OS validation,
 * portable scalar fallback, and environment override testing.
 * ----------------------------------------------------------------------- */

namespace adaptq {
// Backend factory declarations
IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();
}

#if defined(_WIN32)
static void set_env_var(const char *name, const char *val) {
    _putenv_s(name, val);
}
static void unset_env_var(const char *name) {
    _putenv_s(name, "");
}
#else
static void set_env_var(const char *name, const char *val) {
    setenv(name, val, 1);
}
static void unset_env_var(const char *name) {
    unsetenv(name);
}
#endif


namespace {

static std::vector<uint8_t> make_packed(int padded, int bits,
                                        std::mt19937 &rng) {
    const int bytes = (padded * bits + 7) / 8;
    std::vector<uint8_t> data(bytes);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (uint8_t &byte : data)
        byte = static_cast<uint8_t>(byte_dist(rng));
    return data;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float max_diff = 0.f;
    for (int i = 0; i < n; ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    return max_diff;
}

} // namespace

TEST_CASE("Scalar and AVX2 K-dot/V accumulation equivalence",
          "[kernel][avx2][differential]") {
    adaptq::IKernelBackend *scalar = adaptq::create_scalar_backend();
    adaptq::IKernelBackend *avx2 = adaptq::create_avx2_backend();

    REQUIRE(scalar != nullptr);
    if (avx2 == nullptr || !avx2->is_available()) {
        SUCCEED("AVX2 backend is unavailable on this host");
        return;
    }

    std::mt19937 rng(0xA11CE55u);
    std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.1f, 1.5f);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);

    const int padded_dims[] = {8, 16, 64, 128};
    const int batch_sizes[] = {1, 3, 4, 5, 9};

    for (int bits : {2, 3, 4}) {
        for (int padded : padded_dims) {
            std::vector<float> q_rot(padded);
            for (float &value : q_rot)
                value = q_dist(rng);

            for (int n : batch_sizes) {
                const int packed_bytes = (padded * bits + 7) / 8;
                std::vector<std::vector<uint8_t>> k_storage(n);
                std::vector<std::vector<uint8_t>> v_storage(n);
                std::vector<adaptq::CompressResult> k_results(n);
                std::vector<adaptq::CompressResult> v_results(n);
                std::vector<float> weights(n);

                for (int i = 0; i < n; ++i) {
                    k_storage[i] = make_packed(padded, bits, rng);
                    v_storage[i] = make_packed(padded, bits, rng);
                    const float k_scale = scale_dist(rng);
                    const float v_scale = scale_dist(rng);

                    k_results[i] = {
                        k_storage[i].data(), packed_bytes, k_scale, 0, 0
                    };
                    v_results[i] = {
                        v_storage[i].data(), packed_bytes, v_scale, 0, 0
                    };
                    weights[i] = weight_dist(rng);
                }
                weights[0] = 0.0f;
                if (n > 3)
                    weights[3] = 0.0f;

                std::vector<float> scalar_logits(n), avx2_logits(n);
                scalar->kdot_batch(q_rot.data(), k_results.data(), n, padded,
                                   bits, scalar_logits.data());
                avx2->kdot_batch(q_rot.data(), k_results.data(), n, padded,
                                 bits, avx2_logits.data());

                CAPTURE(bits, padded, n);
                REQUIRE(max_abs_diff(scalar_logits.data(), avx2_logits.data(),
                                     n) < 1e-4f);

                std::vector<float> scalar_acc(padded), avx2_acc(padded);
                scalar->vaccum_batch(scalar_acc.data(), v_results.data(),
                                     weights.data(), n, padded, bits);
                avx2->vaccum_batch(avx2_acc.data(), v_results.data(),
                                   weights.data(), n, padded, bits);

                REQUIRE(max_abs_diff(scalar_acc.data(), avx2_acc.data(),
                                     padded) < 2e-4f);
            }
        }
    }
}

TEST_CASE("AVX2 Capability Detection and OS XCR0 State", "[kernel][avx2]") {
    bool has_avx2 = adaptq::cpu_supports_avx2();
    // cpu_supports_avx2 returns a clean boolean without illegal-instruction faults
    INFO("Host CPU + OS AVX2 support status: " << (has_avx2 ? "ENABLED" : "DISABLED/FALLBACK"));
    SUCCEED("cpu_supports_avx2 evaluated cleanly");
}

TEST_CASE("AVX2 Backend Factory Contract", "[kernel][avx2]") {
    adaptq::IKernelBackend *backend = adaptq::create_avx2_backend();

    if (backend) {
        REQUIRE(backend->is_available());
        REQUIRE(std::string(backend->name()) == "avx2");
    } else {
        SUCCEED("AVX2 backend is unavailable in this portable build");
    }
}

TEST_CASE("Scalar Backend Factory and Interface Conformance", "[kernel][scalar]") {
    adaptq::IKernelBackend *scalar_backend = adaptq::create_scalar_backend();
    REQUIRE(scalar_backend != nullptr);
    REQUIRE(scalar_backend->is_available() == true);
    REQUIRE(std::string(scalar_backend->name()) == "scalar");
}

TEST_CASE("Runtime Environment Variable Fallback Override", "[kernel][fallback]") {
    // Ensure clean initial state
    unset_env_var("ADAPTQ_DISABLE_AVX2");
    unset_env_var("ADAPTQ_FORCE_SCALAR");
    unset_env_var("ADAPTQ_BACKEND");

    REQUIRE(adaptq::is_scalar_forced() == false);

    SECTION("Override via ADAPTQ_DISABLE_AVX2=1") {
        set_env_var("ADAPTQ_DISABLE_AVX2", "1");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_DISABLE_AVX2");
    }

    SECTION("Override via ADAPTQ_FORCE_SCALAR=1") {
        set_env_var("ADAPTQ_FORCE_SCALAR", "1");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_FORCE_SCALAR");
    }

    SECTION("Override via ADAPTQ_BACKEND=scalar") {
        set_env_var("ADAPTQ_BACKEND", "scalar");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_BACKEND");
    }

    // Restore clean state
    unset_env_var("ADAPTQ_DISABLE_AVX2");
    unset_env_var("ADAPTQ_FORCE_SCALAR");
    unset_env_var("ADAPTQ_BACKEND");
}
