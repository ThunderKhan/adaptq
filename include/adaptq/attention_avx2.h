#pragma once

bool adaptq_attention_avx2_compute(
    const float *q_rot,
    float *acc,
    const float *codebook,
    const unsigned char *k_data,
    const unsigned char *v_data,
    const float *k_scale,
    const float *v_scale,
    float attention_scale,
    float inverse_sqrt_padded,
    int *slots,
    int token_count,
    int packed_bytes,
    int padded,
    int bits,
    float value_mass_threshold,
    int *order,
    float *logits);
