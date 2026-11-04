"""CPU reference for inspecting PITM forward operators on synthetic TF envelopes."""
import argparse
import json
from pathlib import Path

import numpy as np
from preview import save_preview


ROOT = Path(__file__).resolve().parent
RADII = (1, 3, 5)
MODES = (1, 8, 16)
C_WATER, DEPTH, PATH = 1500.0, 110.0, 18000.0
SLOPE, CUTOFF = 50.0, 200.0
SEED = 20261104


def relative_delay(frequency, mode):
    """Leading differential modal travel time versus mode 1; seconds."""
    return (PATH / (2 * C_WATER) * (C_WATER / (2 * DEPTH * frequency)) ** 2
            * (mode**2 - 1))


def background(rng, frequency, time):
    """Rank-two coherent envelope plus independent compound-Gaussian envelopes."""
    smooth = 0.9 + 0.4 * np.cos(frequency[:, None] / 72)
    band = np.exp(-0.5 * ((frequency[:, None] - 455) / 48) ** 2)
    phases = rng.uniform(0, 2 * np.pi, 2)
    coherent = (smooth * (1 + 0.3 * np.sin(time / 1.5 + phases[0]))
                + band * (1 + 0.4 * np.cos(time / 2.8 + phases[1])))
    shape = (frequency.size, time.size)
    texture = rng.gamma(0.3, 1 / 0.3, size=shape)
    speckle = rng.rayleigh(1 / np.sqrt(2), size=shape)
    return coherent + 0.18 * np.sqrt(texture) * speckle


def generate(destination, seed):
    rng = np.random.default_rng(seed)
    frequency = np.arange(220., 661., 5.)
    time = np.arange(0., 16., 0.1)
    reference_time = time[:60]
    raw, target, reference, centers, widths, present = [], [], [], [], [], []
    for center, width in [(440., 5.), (452., 4.)]:
        clutter = background(rng, frequency, time)
        ref = background(rng, frequency, reference_time)
        lower, upper = center - SLOPE * width / 2, center + SLOPE * width / 2
        band = (frequency >= lower) & (frequency <= upper)
        echo = np.zeros_like(clutter)
        for mode, gain in zip(MODES, (1., 0.75, 0.45)):
            arrival = (4. + (frequency - lower) / SLOPE
                       + relative_delay(frequency, mode))
            echo += gain * band[:, None] * np.exp(
                -0.5 * ((time - arrival[:, None]) / 0.10) ** 2)
        for flag in (False, True):
            target.append(float(flag) * echo)
            raw.append(clutter + target[-1])
            reference.append(ref)
            centers.append(center)
            widths.append(width)
            present.append(flag)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, raw=raw, reference=reference, clean_echo=target,
                        frequency_hz=frequency, time_s=time, reference_time_s=reference_time,
                        center_hz=centers, width_s=widths, target_present=present, seed=seed)


def shrink(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def reference_basis(reference):
    """Principal component pursuit by ADMM with a fixed scalar penalty."""
    penalty = 1 / np.sqrt(max(reference.shape))
    mu = 2.5 / np.mean(np.abs(reference))
    low = np.zeros_like(reference)
    dual = np.zeros_like(reference)
    norm = np.linalg.norm(reference)
    for step in range(1, 1001):
        sparse = shrink(reference - low + dual / mu, penalty / mu)
        left, singular, right = np.linalg.svd(
            reference - sparse + dual / mu, full_matrices=False)
        next_low = (left * np.maximum(singular - 1 / mu, 0)) @ right
        residual = reference - next_low - sparse
        primal = np.linalg.norm(residual) / norm
        change = np.linalg.norm(next_low - low) / norm
        low = next_low
        dual += mu * residual
        if max(primal, change) < 1e-4:
            break
    else:
        raise RuntimeError("RPCA failed to reach example tolerance")
    basis = np.linalg.svd(low, full_matrices=False)[0][:, :2]
    return basis, step, primal, change


def sample_shifted(x, df, dt):
    """Sample x[f+df,t+dt] on a bilinear grid with zero extension."""
    pf = np.arange(x.shape[0])[:, None] + df
    pt = np.arange(x.shape[1])[None, :] + dt
    i, j = np.floor(pf).astype(int), np.floor(pt).astype(int)
    a, b = pf - i, pt - j
    padded = np.pad(x, 1)
    value = np.zeros_like(x)
    for di, wi in [(0, 1 - a), (1, a)]:
        for dj, wj in [(0, 1 - b), (1, b)]:
            ii = np.clip(i + di + 1, 0, x.shape[0] + 1)
            jj = np.clip(j + dj + 1, 0, x.shape[1] + 1)
            value += wi * wj * padded[ii, jj]
    return value


def branch(masked, radius, extent, df_bin, dt_bin, center):
    # Smooth and across-ridge derivative templates; scale axes are in TF bins.
    grid = np.arange(-radius, radius + 1)
    v, u = np.meshgrid(grid / radius, grid / radius, indexing="ij")
    slow = np.exp(-1.5 * (u**2 + v**2) - 3 * (v - u)**2)
    slow /= slow.sum()
    fast = slow * ((v - u)**2 + 0.2)
    fast -= np.sum(fast * slow) / np.sum(slow**2) * slow
    fast /= np.abs(fast).sum()
    dot = abs(np.sum(slow * fast))
    yslow = np.zeros_like(masked)
    yfast = np.zeros_like(masked)
    time_step = extent / (radius * dt_bin)
    freq_step = (5.5 * 440 / center) / df_bin
    for vi, v0 in enumerate(grid):
        for ui, u0 in enumerate(grid):
            sampled = sample_shifted(masked, v0 * freq_step, u0 * time_step)
            yslow += slow[vi, ui] * sampled
            yfast += fast[vi, ui] * sampled
    return yslow + 0.2 * shrink(yfast, 0.10), dot


def forward(raw, reference, frequency, time, center, width):
    basis, steps, primal, change = reference_basis(reference)
    projected = raw - basis @ (basis.T @ raw)
    lower = max(CUTOFF, center - abs(SLOPE) * width / 2)
    upper = center + abs(SLOPE) * width / 2
    band = (frequency >= lower) & (frequency <= upper)
    masked = projected * band[:, None]
    max_delay = float(relative_delay(lower, max(MODES)))
    extent = max_delay * (1 + np.logaddexp(0, [-2., -0.5, 0.5]))
    maps, dots = [], []
    for radius, span in zip(RADII, extent):
        y, dot = branch(masked, radius, span, frequency[1] - frequency[0],
                        time[1] - time[0], center)
        maps.append(y)
        dots.append(dot)
    maps = np.stack(maps)
    logits = np.abs(maps)
    weights = np.exp(logits - logits.max(axis=0))
    weights /= weights.sum(axis=0)
    score = (weights * maps).sum(axis=0) * band[:, None]
    report = dict(band_hz=[lower, upper], delay_bound_s=max_delay,
                  time_half_extents_s=extent.tolist(), admm_steps=steps,
                  admm_primal_relative=primal, admm_change_relative=change,
                  projection_relative=float(np.linalg.norm(basis.T @ projected)
                                            / np.linalg.norm(raw)),
                  max_kernel_inner_product=max(dots),
                  max_gate_sum_error=float(np.max(np.abs(weights.sum(axis=0) - 1))),
                  off_band_mask_abs_max=float(np.max(np.abs(masked[~band]))))
    if not (np.isfinite(score).all() and report['projection_relative'] < 1e-12
            and report['max_kernel_inner_product'] < 1e-12
            and report['max_gate_sum_error'] < 1e-12 and np.all(extent >= max_delay)
            and report['off_band_mask_abs_max'] == 0):
        raise RuntimeError("A forward-operator invariant failed")
    return projected, masked, score, report


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        frequency, time = data['frequency_hz'], data['time_s']
        projected, masked, scores, reports = [], [], [], []
        for i, (raw, ref, center, width) in enumerate(zip(
                data['raw'], data['reference'], data['center_hz'], data['width_s'])):
            p, m, s, report = forward(raw, ref, frequency, time, center, width)
            projected.append(p)
            masked.append(m)
            scores.append(s)
            report.update(case=i, center_hz=float(center), width_s=float(width))
            reports.append(report)
        np.savez_compressed(output / 'forward_maps.npz', projected=projected, masked=masked,
                            response=scores, frequency_hz=frequency, time_s=time)
        pairs = []
        for background_case, echo_case in ((0, 1), (2, 3)):
            # Both observations in a pair share the same clutter and reference.
            # Ground truth is consulted here, after evaluating the forward maps.
            np.testing.assert_allclose(
                data['raw'][echo_case] - data['raw'][background_case],
                data['clean_echo'][echo_case], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(data['reference'][echo_case],
                                          data['reference'][background_case])
            delta = scores[echo_case] - scores[background_case]
            fi, ti = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
            pairs.append(dict(background_case=background_case, echo_case=echo_case,
                              center_hz=reports[echo_case]['center_hz'],
                              width_s=reports[echo_case]['width_s'],
                              peak_abs_change=float(abs(delta[fi, ti])),
                              peak_frequency_hz=float(frequency[fi]),
                              peak_time_s=float(time[ti])))
        save_preview(output / 'paired_response.png', scores, pairs, frequency, time)
        summary = dict(provenance="synthetic operator demonstration", seed=int(data['seed']),
                       cases=reports, paired_responses=pairs)
    (output / 'checks.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"{len(reports)} cases passed. Inspect {output}")
    for pair in pairs:
        print(f"  {pair['center_hz']:g} Hz / {pair['width_s']:g} s: "
              f"peak paired change {pair['peak_abs_change']:.4f} a.u. "
              f"at {pair['peak_frequency_hz']:g} Hz, {pair['peak_time_s']:.1f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'synthetic_tf.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--generate', action='store_true', help='Regenerate --data before running')
    parser.add_argument('--seed', type=int, default=SEED, help='Generator seed')
    args = parser.parse_args()
    if args.generate:
        generate(args.data, args.seed)
    run(args.data, args.output)
