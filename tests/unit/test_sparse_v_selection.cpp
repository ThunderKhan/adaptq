#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq.h"
#include <cmath>
#include <vector>

TEST_CASE("sparse V selection remains finite on a larger cache",
          "[attention][sparse][selection]") {
    adaptq_ctx_t h = adaptq_create(64, 4, 256, 77, 0.9f, 0);
    REQUIRE(h != nullptr);

    std::vector<float> k(64), v(64), q(64), out(64);
    for (int token = 0; token < 128; ++token) {
        for (int i = 0; i < 64; ++i) {
            k[i] = std::sin((float)(i + token) * 0.031f);
            v[i] = std::cos((float)(i * 3 + token) * 0.017f);
        }
        adaptq_append(h, k.data(), v.data(), token);
    }

    for (int i = 0; i < 64; ++i)
        q[i] = std::sin((float)i * 0.043f + 0.7f);

    REQUIRE(adaptq_compute(h, q.data(), out.data()) == 128);
    for (float value : out)
        REQUIRE(std::isfinite(value));

    adaptq_destroy(h);
}
