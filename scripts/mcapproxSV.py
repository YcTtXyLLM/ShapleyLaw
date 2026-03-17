import itertools
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Coalition = Tuple[str, ...]
UtilityMap = Dict[Coalition, float]
ShapleyMap = Dict[str, float]
History = List[Tuple[int, ShapleyMap]]


def normalize_coalitions(utility_raw: Mapping[Iterable[str], float]) -> UtilityMap:
    """
    Normalize coalition keys by converting them to sorted tuples.
    """
    utility: UtilityMap = {}
    for coalition, value in utility_raw.items():
        key = coalition if isinstance(coalition, tuple) else tuple(coalition)
        key = tuple(sorted(key))
        utility[key] = float(value)
    return utility


def shapley_values_exact(
    utility_raw: Mapping[Iterable[str], float],
    players: Sequence[str],
) -> ShapleyMap:
    """
    Compute exact Shapley values from a complete coalition utility table.

    Args:
        utility_raw: Mapping from coalition to utility value.
        players: Sequence of player identifiers.

    Returns:
        Dictionary mapping each player to its exact Shapley value.
    """
    utility = normalize_coalitions(utility_raw)
    players = list(players)
    n = len(players)
    shapley = {p: 0.0 for p in players}
    fact = math.factorial

    for r in range(n + 1):
        for coalition in itertools.combinations(players, r):
            coalition_key = tuple(sorted(coalition))
            if coalition_key not in utility:
                raise ValueError(f"Missing utility for coalition {coalition_key}")
            u_s = utility[coalition_key]

            for player in players:
                if player in coalition:
                    continue

                extended_key = tuple(sorted(coalition + (player,)))
                if extended_key not in utility:
                    raise ValueError(f"Missing utility for coalition {extended_key}")

                u_si = utility[extended_key]
                weight = fact(len(coalition)) * fact(n - len(coalition) - 1) / fact(n)
                shapley[player] += weight * (u_si - u_s)

    return shapley


def monte_carlo_shapley(
    utility_raw: Mapping[Iterable[str], float],
    players: Sequence[str],
    num_samples: int = 10000,
    seed: Optional[int] = 42,
    return_history: bool = False,
) -> Union[ShapleyMap, Tuple[ShapleyMap, History]]:
    """
    Estimate Shapley values via Monte Carlo permutation sampling.

    Args:
        utility_raw: Mapping from coalition to utility value.
        players: Sequence of player identifiers.
        num_samples: Number of random permutations.
        seed: Random seed.
        return_history: Whether to return intermediate estimates.

    Returns:
        Either:
            - shapley estimates, or
            - (shapley estimates, history) if return_history=True
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    utility = normalize_coalitions(utility_raw)
    players = list(players)
    rng = random.Random(seed)

    missing = []
    for r in range(len(players) + 1):
        for coalition in itertools.combinations(players, r):
            key = tuple(sorted(coalition))
            if key not in utility:
                missing.append(key)

    if missing:
        preview = ", ".join(map(str, missing[:5]))
        more = " ..." if len(missing) > 5 else ""
        raise ValueError(
            "Monte Carlo estimation requires coalition utilities for lookups. "
            f"Missing {len(missing)} coalitions: {preview}{more}"
        )

    shapley_sum = {p: 0.0 for p in players}
    history: History = []
    report_every = 1000 if num_samples >= 1000 else 1

    for t in range(1, num_samples + 1):
        permutation = players[:]
        rng.shuffle(permutation)

        current_coalition: List[str] = []
        previous_key: Coalition = ()
        previous_utility = utility[previous_key]

        for player in permutation:
            current_coalition.append(player)
            current_key = tuple(sorted(current_coalition))
            current_utility = utility[current_key]

            shapley_sum[player] += (current_utility - previous_utility)

            previous_key = current_key
            previous_utility = current_utility

        if return_history and (t % report_every == 0 or t == num_samples):
            history.append((t, {p: shapley_sum[p] / t for p in players}))

    shapley = {p: shapley_sum[p] / num_samples for p in players}

    if return_history:
        return shapley, history
    return shapley


def print_shapley(title: str, shapley: Mapping[str, float]) -> None:
    """
    Print Shapley values in a compact format.
    """
    print(title)
    for player, value in sorted(shapley.items()):
        print(f"  {player}: {value:.6f}")


def compare_exact_and_mc(
    utility_raw: Mapping[Iterable[str], float],
    players: Sequence[str],
    num_samples: int = 10000,
    seed: Optional[int] = 42,
) -> None:
    """
    Compare exact and Monte Carlo Shapley values.
    """
    exact = shapley_values_exact(utility_raw, players)
    approx = monte_carlo_shapley(
        utility_raw,
        players,
        num_samples=num_samples,
        seed=seed,
        return_history=False,
    )

    print(f"Comparison with {num_samples:,} Monte Carlo samples")
    print("player        exact           approx          abs_error")
    print("-" * 60)
    for player in sorted(players):
        error = abs(exact[player] - approx[player])
        print(f"{player:<8} {exact[player]:>12.6f} {approx[player]:>15.6f} {error:>14.6f}")


def run_experiments(
    data: Mapping[str, Mapping[Iterable[str], float]],
    num_samples: int = 10000,
    seed: Optional[int] = 42,
    compute_exact: bool = True,
    negate_scores: bool = False,
) -> None:
    """
    Run Shapley-value analysis for multiple experimental groups.

    Args:
        data: Mapping from group identifier to coalition-score mapping.
        num_samples: Number of Monte Carlo samples.
        seed: Random seed.
        compute_exact: Whether to also compute exact Shapley values.
        negate_scores: If True, transform each score by negation
            (useful when the provided values are losses instead of utilities).
    """
    for group_id, raw_scores in data.items():
        print(f"\n=== Results for {group_id} ===")

        players = sorted({player for coalition in raw_scores for player in coalition})
        print(f"Players: {players}")

        utility_map = {
            tuple(sorted(coalition)): (-score if negate_scores else score)
            for coalition, score in raw_scores.items()
        }

        mc_shapley = monte_carlo_shapley(
            utility_map,
            players,
            num_samples=num_samples,
            seed=seed,
            return_history=False,
        )

        print_shapley(
            f"Monte Carlo Shapley values (samples={num_samples:,}, seed={seed}):",
            mc_shapley,
        )

        if compute_exact:
            exact_shapley = shapley_values_exact(utility_map, players)
            print_shapley("Exact Shapley values:", exact_shapley)

            print("Absolute errors:")
            for player in sorted(players):
                error = abs(exact_shapley[player] - mc_shapley[player])
                print(f"  {player}: {error:.6f}")


if __name__ == "__main__":
    # Example placeholder.
    # Replace this with the actual coalition-score table used in the experiment.
    data = {
        "group_1": {
            (): 0.0,
            ("A",): 1.2,
            ("B",): 0.9,
            ("C",): 1.5,
            ("A", "B"): 2.0,
            ("A", "C"): 2.8,
            ("B", "C"): 2.4,
            ("A", "B", "C"): 3.1,
        }
    }

    run_experiments(
        data=data,
        num_samples=1000,
        seed=42,
        compute_exact=True,
        negate_scores=False,
    )
