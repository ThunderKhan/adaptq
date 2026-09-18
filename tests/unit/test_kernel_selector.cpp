/* Catch2 v3 — scalar/AVX2 kernel differential coverage */
#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq/kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace adaptq {
IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();
}

namespace {

std::vector<uint8_t> make_packed(int padded, int bits, std::mt19937 &rng) {
    const int bytes = (padded * bits + 7) / 8;
    std::vector<uint8_t> packed(bytes);
    std::uniform_int_distribution<int> dist(0, 255);
    for (uint8_t &byte : packed)
        byte = static_cast<uint8_t>(dist(rng));
    return packed;
}

float max_abs_diff(const float *a, const float *b, int n) {
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
    std::uniform_real_distribution<float> q_dist(-1.f, 1.f);
    std::uniform_real_distribution<float> scale_dist(0.1f, 1.5f);
    std::uniform_real_distribution<float> weight_dist(-1.f, 1.f);

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

                    k_results[i] = {
                        k_storage[i].data(), packed_bytes, scale_dist(rng), 0, 0
                    };
                    v_results[i] = {
                        v_storage[i].data(), packed_bytes, scale_dist(rng), 0, 0
                    };
                    weights[i] = weight_dist(rng);
                }

                std::vector<float> scalar_logits(n);
                std::vector<float> avx2_logits(n);
                scalar->kdot_batch(q_rot.data(), k_results.data(), n, padded,
                                   bits, scalar_logits.data());
                avx2->kdot_batch(q_rot.data(), k_results.data(), n, padded,
                                 bits, avx2_logits.data());

                CAPTURE(bits, padded, n);
                REQUIRE(max_abs_diff(scalar_logits.data(), avx2_logits.data(),
                                     n) < 1e-4f);

                std::vector<float> scalar_acc(padded);
                std::vector<float> avx2_acc(padded);
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
