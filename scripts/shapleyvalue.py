import itertools
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Coalition = Tuple[str, ...]
UtilityMap = Dict[Coalition, float]
ShapleyMap = Dict[str, float]


def normalize_coalitions(utility_raw: Mapping[Iterable[str], float]) -> UtilityMap:
    """
    Normalize coalition keys by converting them to sorted tuples.

    Args:
        utility_raw: Mapping from coalition to utility value.

    Returns:
        A dictionary with sorted tuple keys.
    """
    utility: UtilityMap = {}
    for coalition, value in utility_raw.items():
        key = coalition if isinstance(coalition, tuple) else tuple(coalition)
        key = tuple(sorted(key))
        utility[key] = float(value)
    return utility


def shapley_values(
    utility_raw: Mapping[Iterable[str], float],
    players: Sequence[str],
) -> ShapleyMap:
    """
    Compute standard Shapley values for a complete coalition utility table.

    Args:
        utility_raw: Mapping from coalition to utility value.
        players: Sequence of player identifiers.

    Returns:
        Dictionary mapping each player to its Shapley value.
    """
    utility = normalize_coalitions(utility_raw)
    players = list(players)
    n = len(players)
    shapley = {player: 0.0 for player in players}
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

                weight = (
                    fact(len(coalition))
                    * fact(n - len(coalition) - 1)
                    / fact(n)
                )

                shapley[player] += weight * (u_si - u_s)

    return shapley


def print_shapley(title: str, shapley: Mapping[str, float]) -> None:
    """
    Print Shapley values in a compact format.
    """
    print(title)
    for player, value in sorted(shapley.items()):
        print(f"  {player}: {value:.6f}")


def run_shapley_analysis(
    data: Mapping[str, Mapping[Iterable[str], float]],
    negate_scores: bool = False,
) -> None:
    """
    Run Shapley value analysis for multiple groups.

    Args:
        data: Mapping from group identifier to coalition-score mapping.
        negate_scores: If True, negate each score (e.g., convert loss to utility).
    """
    for group_id, raw_scores in data.items():
        print(f"\n=== Results for {group_id} ===")

        players = sorted({player for coalition in raw_scores for player in coalition})
        print(f"Players: {players}")

        utility_map = {
            tuple(sorted(coalition)): (-score if negate_scores else score)
            for coalition, score in raw_scores.items()
        }

        shapley = shapley_values(utility_map, players)

        print_shapley("Shapley values:", shapley)


if __name__ == "__main__":
    # Example placeholder (replace with actual data used in experiments)
    data = {
        "group_1": {
            (): 0.0,
            ("A",): 1.0,
            ("B",): 1.2,
            ("C",): 0.8,
            ("A", "B"): 2.5,
            ("A", "C"): 2.0,
            ("B", "C"): 2.3,
            ("A", "B", "C"): 3.1,
        }
    }

    run_shapley_analysis(
        data=data,
        negate_scores=False,
    )
