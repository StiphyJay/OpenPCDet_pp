import torch


def gen_pyramid_xyz(features, scale=0.01):
    """
    features shape [n_points,3]
    return 3 8-bit simulated quantized features with shape [n_points,3]
    """
    # int24 quantization on features

    scale_a = scale * (1 << 15)
    scale_b = scale * (1 << 7)
    scale_c = scale
    qmin = -128
    qmax = 127
    feature_a_int = (features / scale_a).round().clamp(qmin, qmax)
    feature_a_sim = feature_a_int * scale_a
    feature_b_int = ((features - feature_a_sim) / scale_b).round().clamp(qmin, qmax)
    feature_b_sim = feature_b_int * scale_b
    feature_c_int = ((features - feature_a_sim - feature_b_sim) / scale_c).round().clamp(qmin, qmax)
    feature_c_sim = feature_c_int * scale_c
    return feature_a_sim, feature_b_sim, feature_c_sim
