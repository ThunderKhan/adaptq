#include "../include/adaptq/kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace adaptq {
IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();
}

namespace {

struct Options {
    int iterations = 2000;
    int warmup = 200;
    int tokens = 128;
    int padded = 128;
    std::string output;
};

struct Dataset {
    int bits;
    int padded;
    int tokens;
    std::vector<float> q_rot;
    std::vector<std::vector<uint8_t>> k_storage;
    std::vector<std::vector<uint8_t>> v_storage;
    std::vector<adaptq::CompressResult> k_results;
    std::vector<adaptq::CompressResult> v_results;
    std::vector<float> weights;
};

struct Result {
    std::string workload;
    std::string backend;
    int bits;
    int tokens;
    int padded;
    int iterations;
    double ns_per_iteration;
    double checksum;
    double speedup_vs_scalar;
};

volatile float benchmark_sink = 0.f;

void usage(const char *program) {
    std::cout
        << "Usage: " << program
        << " [--iterations N] [--warmup N] [--tokens N] [--padded N]"
           " [--output PATH]\n";
}

bool parse_int(const std::string &value, int &out) {
    try {
        size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed <= 0)
            return false;
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char **argv, Options &options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return false;
        }

        auto next_value = [&](int &value) {
            if (i + 1 >= argc)
                return false;
            return parse_int(argv[++i], value);
        };

        if (arg == "--iterations") {
            if (!next_value(options.iterations))
                return false;
        } else if (arg == "--warmup") {
            if (!next_value(options.warmup))
                return false;
        } else if (arg == "--tokens") {
            if (!next_value(options.tokens))
                return false;
        } else if (arg == "--padded") {
            if (!next_value(options.padded))
                return false;
        } else if (arg == "--output") {
            if (i + 1 >= argc)
                return false;
            options.output = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
            return false;
        }
    }

    if ((options.padded & (options.padded - 1)) != 0) {
        std::cerr << "--padded must be a positive power of two\n";
        return false;
    }
    if (options.tokens <= 0) {
        std::cerr << "--tokens must be positive\n";
        return false;
    }
    return true;
}

std::vector<uint8_t> make_packed(int padded, int bits, std::mt19937 &rng) {
    const int bytes = (padded * bits + 7) / 8;
    std::vector<uint8_t> data(bytes);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (uint8_t &byte : data)
        byte = static_cast<uint8_t>(byte_dist(rng));
    return data;
}

Dataset make_dataset(int bits, const Options &options, std::mt19937 &rng) {
    Dataset data;
    data.bits = bits;
    data.padded = options.padded;
    data.tokens = options.tokens;
    data.q_rot.resize(options.padded);
    data.k_storage.resize(options.tokens);
    data.v_storage.resize(options.tokens);
    data.k_results.resize(options.tokens);
    data.v_results.resize(options.tokens);
    data.weights.resize(options.tokens);

    std::uniform_real_distribution<float> q_dist(-1.f, 1.f);
    std::uniform_real_distribution<float> scale_dist(0.1f, 1.5f);
    std::uniform_real_distribution<float> weight_dist(0.0f, 1.0f);

    for (float &value : data.q_rot)
        value = q_dist(rng);

    const int packed_bytes = (options.padded * bits + 7) / 8;
    for (int i = 0; i < options.tokens; ++i) {
        data.k_storage[i] = make_packed(options.padded, bits, rng);
        data.v_storage[i] = make_packed(options.padded, bits, rng);
        data.k_results[i] = {
            data.k_storage[i].data(), packed_bytes, scale_dist(rng), 0, 0
        };
        data.v_results[i] = {
            data.v_storage[i].data(), packed_bytes, scale_dist(rng), 0, 0
        };
        data.weights[i] = weight_dist(rng);
    }

    return data;
}

void softmax(float *values, int n) {
    float max_value = values[0];
    for (int i = 1; i < n; ++i)
        max_value = std::max(max_value, values[i]);

    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        values[i] = std::exp(values[i] - max_value);
        sum += values[i];
    }

    const float inv_sum = 1.f / sum;
    for (int i = 0; i < n; ++i)
        values[i] *= inv_sum;
}

template <typename Fn>
double measure(Fn &&fn, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i)
        fn();

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        fn();
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count()
           / static_cast<double>(iterations);
}

double checksum(const std::vector<float> &values) {
    double sum = 0.0;
    for (float value : values)
        sum += static_cast<double>(value);
    return sum;
}

Result run_kdot(adaptq::IKernelBackend *backend, Dataset &data,
                const Options &options) {
    std::vector<float> logits(data.tokens);
    const double ns = measure(
        [&] {
            backend->kdot_batch(data.q_rot.data(), data.k_results.data(),
                                data.tokens, data.padded, data.bits,
                                logits.data());
        },
        options.warmup, options.iterations);

    const double sum = checksum(logits);
    benchmark_sink = logits[0];
    return {"kdot", backend->name(), data.bits, data.tokens, data.padded,
            options.iterations, ns, sum, 0.0};
}

Result run_vaccum(adaptq::IKernelBackend *backend, Dataset &data,
                  const Options &options) {
    std::vector<float> acc(data.padded);
    const double ns = measure(
        [&] {
            backend->vaccum_batch(acc.data(), data.v_results.data(),
                                  data.weights.data(), data.tokens,
                                  data.padded, data.bits);
        },
        options.warmup, options.iterations);

    const double sum = checksum(acc);
    benchmark_sink = acc[0];
    return {"vaccum", backend->name(), data.bits, data.tokens, data.padded,
            options.iterations, ns, sum, 0.0};
}

Result run_attention(adaptq::IKernelBackend *backend, Dataset &data,
                     const Options &options) {
    std::vector<float> logits(data.tokens);
    std::vector<float> acc(data.padded);

    const double ns = measure(
        [&] {
            backend->kdot_batch(data.q_rot.data(), data.k_results.data(),
                                data.tokens, data.padded, data.bits,
                                logits.data());
            softmax(logits.data(), data.tokens);
            backend->vaccum_batch(acc.data(), data.v_results.data(),
                                  logits.data(), data.tokens,
                                  data.padded, data.bits);
        },
        options.warmup, options.iterations);

    const double sum = checksum(acc);
    benchmark_sink = acc[0];
    return {"attention", backend->name(), data.bits, data.tokens, data.padded,
            options.iterations, ns, sum, 0.0};
}

void print_csv_header(std::ostream &out) {
    out << "workload,backend,bits,tokens,padded,iterations,"
           "ns_per_iteration,checksum,speedup_vs_scalar\n";
}

void print_csv_row(std::ostream &out, const Result &result) {
    out << result.workload << ','
        << result.backend << ','
        << result.bits << ','
        << result.tokens << ','
        << result.padded << ','
        << result.iterations << ','
        << std::fixed << std::setprecision(3)
        << result.ns_per_iteration << ','
        << std::setprecision(9)
        << result.checksum << ',';
    if (result.speedup_vs_scalar > 0.0)
        out << std::setprecision(3) << result.speedup_vs_scalar;
    out << '\n';
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_args(argc, argv, options))
        return 2;

    adaptq::IKernelBackend *scalar = adaptq::create_scalar_backend();
    adaptq::IKernelBackend *avx2 = adaptq::create_avx2_backend();

    if (scalar == nullptr || avx2 == nullptr || !avx2->is_available()) {
        std::cerr << "AVX2 backend is unavailable; native AVX2 benchmark cannot run.\n";
        return 2;
    }

    std::mt19937 rng(0xBADC0DEu);
    std::vector<Result> results;

    for (int bits : {2, 3, 4}) {
        Dataset data = make_dataset(bits, options, rng);

        Result scalar_kdot = run_kdot(scalar, data, options);
        Result avx2_kdot = run_kdot(avx2, data, options);
        scalar_kdot.speedup_vs_scalar = 1.0;
        avx2_kdot.speedup_vs_scalar =
            scalar_kdot.ns_per_iteration / avx2_kdot.ns_per_iteration;

        Result scalar_vaccum = run_vaccum(scalar, data, options);
        Result avx2_vaccum = run_vaccum(avx2, data, options);
        scalar_vaccum.speedup_vs_scalar = 1.0;
        avx2_vaccum.speedup_vs_scalar =
            scalar_vaccum.ns_per_iteration / avx2_vaccum.ns_per_iteration;

        Result scalar_attention = run_attention(scalar, data, options);
        Result avx2_attention = run_attention(avx2, data, options);
        scalar_attention.speedup_vs_scalar = 1.0;
        avx2_attention.speedup_vs_scalar =
            scalar_attention.ns_per_iteration / avx2_attention.ns_per_iteration;

        results.push_back(scalar_kdot);
        results.push_back(avx2_kdot);
        results.push_back(scalar_vaccum);
        results.push_back(avx2_vaccum);
        results.push_back(scalar_attention);
        results.push_back(avx2_attention);
    }

    std::ostringstream csv;
    print_csv_header(csv);
    for (const Result &result : results)
        print_csv_row(csv, result);

    std::cout << csv.str();

    if (!options.output.empty()) {
        std::ofstream file(options.output);
        if (!file) {
            std::cerr << "Failed to open output file: " << options.output << "\n";
            return 2;
        }
        file << csv.str();
    }

    return 0;
}
