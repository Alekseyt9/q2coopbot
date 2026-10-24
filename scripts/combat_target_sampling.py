"""CUDA target-factor calibration; the raw actor and conditional branches stay intact."""
import math
from combat_target_head import distribution

VERSION = 'combat_target_temperature_v1'


def validate(header):
    if header is None:
        return None
    if not isinstance(header, dict) or set(header) != {'version', 'temperature'}:
        raise ValueError('Invalid target sampling header')
    value = header['temperature']
    if header['version'] != VERSION or type(value) not in (int, float) or not math.isfinite(value) or not .25 <= value <= 2:
        raise ValueError('Invalid target sampling temperature')
    return dict(header)


def correction(raw, features, target, header):
    header = validate(header)
    if header is None or header['temperature'] == 1:
        return None
    if not raw.is_cuda or not features.is_cuda or not target.is_cuda:
        raise ValueError('CUDA required for target sampling calibration')
    baseline = distribution(raw[:, :45], features)
    scoring = raw[:, :45].clone()
    scoring[:, 20:29] /= header['temperature']
    calibrated = distribution(scoring, features)
    return calibrated.log_prob(target) - baseline.log_prob(target), calibrated.entropy() - baseline.entropy()
