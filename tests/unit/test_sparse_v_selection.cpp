#include <catch2/catch_test_macros.hpp>
#include "../../include/attention.h"
#include "../../include/adaptq/kernel.h"
#include "../../include/fwht.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace adaptq {
IKernelBackend *create_scalar_backend();
}

using namespace adaptq;

static std::vector<float> scalar_sparse_reference(const AttentionHead &head,
                                                  const std::vector<float> &query) {
    const int n = head.kv_buf.size;
    const int cap = head.kv_buf.capacity;
    const int pb = head.kv_buf.packed_bytes;
    const int padded = head.padded;
    const int bits = head.bits;

    std::vector<float> qr(padded, 0.f);
    std::copy(query.begin(), query.end(), qr.begin());

    float qn = 0.f;
    for (float x : query)
        qn += x * x;
    qn = std::sqrt(qn + 1e-12f);
    const float inv_qn = 1.f / qn;
    for (float &x : qr)
        x *= inv_qn;

    fwht_forward(qr.data(), head.quant.D.data(), padded);
    const float sp = std::sqrt((float)padded);
    for (float &x : qr)
        x *= sp * qn;

    std::vector<CompressResult> k_results(n);
    std::vector<CompressResult> v_results(n);
    for (int i = 0; i < n; ++i) {
        const int slot = (head.kv_buf.head - n + cap + i) % cap;
        k_results[i] = {
            head.kv_buf.k_ptr(slot), pb, head.kv_buf.k_scale[slot], 0, (StorageSlot)slot
        };
        v_results[i] = {
            head.kv_buf.v_ptr(slot), pb, head.kv_buf.v_scale[slot], 0, (StorageSlot)slot
        };
    }

    IKernelBackend *scalar = create_scalar_backend();
    std::vector<float> logits(n);
    scalar->kdot_batch(qr.data(), k_results.data(), n, padded, bits, logits.data());
    softmax(logits.data(), n);

    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });

    const float isp = 1.f / std::sqrt((float)padded);
    std::vector<CompressResult> selected_v;
    std::vector<float> weights;
    float mass = 0.f;
    for (int i = 0; i < n && mass < head.v_mass_thresh; ++i) {
        const int idx = ord[i];
        const int slot = (head.kv_buf.head - n + cap + idx) % cap;
        selected_v.push_back(v_results[idx]);
        weights.push_back(logits[idx] * isp);
        mass += logits[idx];
    }

    std::vector<float> acc(padded, 0.f);
    scalar->vaccum_batch(acc.data(), selected_v.data(), weights.data(),
                         (int)selected_v.size(), padded, bits);
    fwht_inverse(acc.data(), head.quant.D.data(), padded);
    acc.resize(head.dim);
    return acc;
}

TEST_CASE("AVX2 sparse V selection matches full-sort reference",
          "[attention][sparse][selection]") {
    AttentionHead head;
    head.init(64, 4, 64, 987, 0.75f, 0);

    std::vector<float> k(64), v(64), q(64);
    for (int token = 0; token < 48; ++token) {
        for (int i = 0; i < 64; ++i) {
            k[i] = std::sin((float)(i * 17 + token * 13) * 0.019f)
                 + 0.2f * std::cos((float)(i + token) * 0.071f);
            v[i] = std::cos((float)(i * 11 - token * 7) * 0.023f)
                 - 0.15f * std::sin((float)(i + token * 3) * 0.059f);
        }
        head.append_kv(k.data(), v.data(), token);
    }
    for (int i = 0; i < 64; ++i)
        q[i] = std::sin((float)i * 0.037f + 0.9f);

    const std::vector<float> expected = scalar_sparse_reference(head, q);
    std::vector<float> actual(64);
    REQUIRE(head.compute(q.data(), actual.data()) == 48);

    for (int i = 0; i < 64; ++i)
        REQUIRE(std::abs(actual[i] - expected[i]) < 2e-3f);
}
