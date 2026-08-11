"""Numba fast path for the fixed-label sector-preserving chain.

Track B needs the true `q_top` at `m = 2, 3`, which exp101 computes by
full-sector thermodynamic integration: one independent chain per logical sector,
annealed along a `K_p` grid. That is the right instrument here precisely because
it never has to transport between sectors -- the barrier appears as a free-energy
difference between independently computed chains rather than as something a
trajectory must cross. But `exp101/src/sector_ti.py` is a pure-Python reference,
and at `m = 3` it costs about twenty hours per disorder.

This module replaces the inner loop and nothing else. It is **bit-exact** with
the certified reference, not merely statistically compatible, because
`_run_label_integrations` already draws from `PortablePrng` -- the same portable
xorshift128+ twin that makes `fast_mcmc.py` bit-identical to `reference_mcmc.py`.
The kernel threads the identical state through the identical sequence of draws:
one Fisher-Yates permutation per sweep, then one `random()` per attempt, drawn
unconditionally even when the move is accepted on energy alone.

Two things are precomputed rather than recomputed per attempt, and both are
identities rather than approximations:

- the qubit support of each proposal, flattened to CSR;
- the set of checks whose syndrome parity flips under that proposal. Whether a
  check flips depends only on `H` and the support, never on the state, so the
  reference's per-attempt `np.unique(np.concatenate(...))` and parity loop
  compute a constant. Hoisting it is what buys most of the speedup.

Everything downstream of the chain -- integration, bootstrap, grid-refinement
gates, sector weights, `q_top` -- stays the certified exp101 code, reached
through `fast_chain_installed()`, which substitutes only `_run_fixed_sector_chain`
and restores it afterwards. Without numba the substitution is a no-op and the
reference runs, correctly and slowly.
"""

from contextlib import contextmanager

import numpy as np

from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101


load_exp101()

try:
    from numba import njit

    from exp101_certified_src.prng import nb_random, nb_randbelow

    NUMBA_AVAILABLE = nb_random is not None
except ImportError:  # pragma: no cover - numba is optional
    njit = None
    NUMBA_AVAILABLE = False


def _exp101_sector_ti():
    from exp101_certified_src import sector_ti

    return sector_ti


def build_kernel_data(model, proposals, q_zero):
    """Flatten the sector-preserving proposals for the kernel.

    Returns CSR arrays for the qubit supports and for the checks whose parity
    the proposal flips, in the reference's own proposal order and restricted to
    the same `usable` subset the reference would use.
    """
    supports = proposals["supports"]
    kinds = proposals["kinds"]
    if q_zero:
        usable = [
            support for support, kind in zip(supports, kinds) if kind == "stab"
        ]
    else:
        usable = list(supports)
    if not usable:
        raise ValueError("no sector-preserving proposals available")

    H = np.asarray(model.H_check, dtype=np.uint8) & np.uint8(1)
    qubit_offsets = np.zeros(len(usable) + 1, dtype=np.int64)
    check_offsets = np.zeros(len(usable) + 1, dtype=np.int64)
    qubit_index = []
    check_index = []
    for position, support in enumerate(usable):
        support = np.asarray(support, dtype=np.int64)
        qubit_index.extend(int(value) for value in support)
        qubit_offsets[position + 1] = len(qubit_index)
        if q_zero:
            # At q=0 the syndrome term is a hard constraint and stabilizer
            # moves never change it, so there is nothing to flip.
            flips = np.zeros(0, dtype=np.int64)
        else:
            parity = np.zeros(H.shape[0], dtype=np.uint8)
            for qubit in support:
                parity ^= H[:, int(qubit)]
            flips = np.flatnonzero(parity).astype(np.int64)
        check_index.extend(int(value) for value in flips)
        check_offsets[position + 1] = len(check_index)
    return {
        "count": len(usable),
        "qubit_offsets": qubit_offsets,
        "qubit_index": np.asarray(qubit_index, dtype=np.int64),
        "check_offsets": check_offsets,
        "check_index": np.asarray(check_index, dtype=np.int64),
    }


def block_bounds(num_measurements, block_count):
    """Reproduce numpy.array_split's block boundaries exactly."""
    base, remainder = divmod(int(num_measurements), int(block_count))
    bounds = np.zeros(int(block_count) + 1, dtype=np.int64)
    position = 0
    for index in range(int(block_count)):
        position += base + (1 if index < remainder else 0)
        bounds[index + 1] = position
    return bounds


if NUMBA_AVAILABLE:

    # Deliberately uncached. exp101's own suite imports the same prng source
    # under the module name `src.prng`, while the bridge imports it as
    # `exp101_certified_src.prng`; two module names for one source collide in a
    # shared numba cache directory and make exp101's bit-exactness tests fail
    # for a reason that has nothing to do with either package. Compilation costs
    # a second or two, against an anchor cell that runs for minutes.
    @njit(cache=False)
    def _kernel(state, v, syndrome_term, data_weight, syndrome_weight,
                qubit_offsets, qubit_index, check_offsets, check_index,
                count, kp_grid, K_q, burn_in, num_measurements,
                sweeps_between, bounds, mu, syndrome_mu, block_mu,
                acceptance, order, samples, syn_samples):
        num_grid = kp_grid.shape[0]
        block_count = bounds.shape[0] - 1
        for grid_index in range(num_grid):
            kp_value = kp_grid[grid_index]
            accepted_count = 0
            attempted_count = 0
            measurement_index = 0
            sweep_in_measurement = 0
            total_sweeps = burn_in + num_measurements * sweeps_between
            for sweep in range(total_sweeps):
                for i in range(count):
                    order[i] = i
                for i in range(count - 1, 0, -1):
                    j = nb_randbelow(state, i + 1)
                    tmp = order[i]
                    order[i] = order[j]
                    order[j] = tmp
                for position in range(count):
                    proposal = order[position]
                    attempted_count += 1
                    q_start = qubit_offsets[proposal]
                    q_stop = qubit_offsets[proposal + 1]
                    overlap = 0
                    for index in range(q_start, q_stop):
                        overlap += v[qubit_index[index]]
                    delta_data = (q_stop - q_start) - 2 * overlap
                    c_start = check_offsets[proposal]
                    c_stop = check_offsets[proposal + 1]
                    delta_syn = 0
                    for index in range(c_start, c_stop):
                        delta_syn += 1 - 2 * syndrome_term[check_index[index]]
                    log_acc = -kp_value * delta_data - K_q * delta_syn
                    # Drawn unconditionally, exactly as the reference does, so
                    # the two streams stay aligned even when the move is
                    # accepted on energy alone.
                    u = nb_random(state)
                    accepted = log_acc >= 0.0 or u < np.exp(log_acc)
                    if accepted:
                        for index in range(q_start, q_stop):
                            v[qubit_index[index]] ^= 1
                        data_weight += delta_data
                        for index in range(c_start, c_stop):
                            check = check_index[index]
                            syndrome_weight += 1 - 2 * syndrome_term[check]
                            syndrome_term[check] ^= 1
                        accepted_count += 1
                if sweep >= burn_in:
                    sweep_in_measurement += 1
                    if sweep_in_measurement == sweeps_between:
                        samples[measurement_index] = data_weight
                        syn_samples[measurement_index] = syndrome_weight
                        measurement_index += 1
                        sweep_in_measurement = 0

            total = 0.0
            syn_total = 0.0
            for index in range(num_measurements):
                total += samples[index]
                syn_total += syn_samples[index]
            mu[grid_index] = total / num_measurements
            syndrome_mu[grid_index] = syn_total / num_measurements
            for block in range(block_count):
                start = bounds[block]
                stop = bounds[block + 1]
                block_total = 0.0
                for index in range(start, stop):
                    block_total += samples[index]
                block_mu[grid_index, block] = block_total / (stop - start)
            if attempted_count > 0:
                acceptance[grid_index] = accepted_count / attempted_count
            else:
                acceptance[grid_index] = 0.0
        return data_weight, syndrome_weight

else:  # pragma: no cover - numba is optional
    _kernel = None


def run_fixed_sector_chain_fast(model, wiring, proposals, v0, kp_grid, config,
                                rng):
    """Drop-in replacement for exp101's `_run_fixed_sector_chain`.

    Same signature, same returned keys, same consumption of `rng`. The state is
    read out of the `PortablePrng`, threaded through the kernel, and written
    back, so a caller that alternates fast and reference chains still sees one
    continuous stream.
    """
    sector_ti = _exp101_sector_ti()
    if _kernel is None:  # pragma: no cover - exercised only without numba
        return sector_ti._run_fixed_sector_chain(
            model, wiring, proposals, v0, kp_grid, config, rng,
        )

    from exp101_certified_src.gf2 import gf2_matmul

    v = np.ascontiguousarray(v0, dtype=np.uint8).copy()
    syndrome_term = (
        gf2_matmul(model.H_check, v[:, None])[:, 0]
        ^ wiring.gibbs_syndrome_argument
    ).astype(np.uint8)
    data_weight = int(v.sum())
    syndrome_weight = int(syndrome_term.sum())
    if wiring.q_zero and syndrome_weight:
        raise AssertionError("q=0 sector representative violates constraint")
    K_q = 0.0 if wiring.q_zero else float(wiring.K_q)

    data = build_kernel_data(model, proposals, bool(wiring.q_zero))
    num_grid = int(len(kp_grid))
    num_measurements = int(config.num_measurements)
    bounds = block_bounds(num_measurements, int(config.block_count))

    mu = np.zeros(num_grid, dtype=np.float64)
    syndrome_mu = np.zeros(num_grid, dtype=np.float64)
    block_mu = np.zeros((num_grid, int(config.block_count)), dtype=np.float64)
    acceptance = np.zeros(num_grid, dtype=np.float64)
    order = np.zeros(data["count"], dtype=np.int64)
    samples = np.zeros(num_measurements, dtype=np.float64)
    syn_samples = np.zeros(num_measurements, dtype=np.float64)

    state = np.array([rng.s0, rng.s1], dtype=np.uint64)
    _kernel(
        state, v, syndrome_term, data_weight, syndrome_weight,
        data["qubit_offsets"], data["qubit_index"],
        data["check_offsets"], data["check_index"], int(data["count"]),
        np.ascontiguousarray(kp_grid, dtype=np.float64), K_q,
        int(config.num_burn_in_sweeps), num_measurements,
        int(config.num_sweeps_between_measurements), bounds,
        mu, syndrome_mu, block_mu, acceptance, order, samples, syn_samples,
    )
    rng.s0 = int(state[0])
    rng.s1 = int(state[1])
    return {"mu": mu, "syndrome_mu": syndrome_mu, "block_mu": block_mu,
            "acceptance": acceptance}


@contextmanager
def fast_chain_installed():
    """Substitute only the inner chain, for the duration of the block.

    Everything else exp101 does -- integration, bootstrap, the coarse/fine grid
    gates, the sector weights, `q_top` and its bootstrap error -- runs as the
    certified code, unmodified. Without numba this yields without substituting
    anything.
    """
    sector_ti = _exp101_sector_ti()
    if _kernel is None:  # pragma: no cover - exercised only without numba
        yield False
        return
    original = sector_ti._run_fixed_sector_chain
    sector_ti._run_fixed_sector_chain = run_fixed_sector_chain_fast
    try:
        yield True
    finally:
        sector_ti._run_fixed_sector_chain = original
