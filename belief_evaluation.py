"""Offline replay and evaluation. Ground truth never enters inference.

Run: python belief_evaluation.py examples/five_player.json --output output/evaluation.json
The JSON is our explicit replay schema, not the raw AIWolf server log format.
Metrics use the final snapshot and exclude the observer's already known role.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from math import log
from pathlib import Path
from time import perf_counter

from belief_model import (ExplainableRoleEstimator, LikelihoodEvidence,
                          LikelihoodParameters, Observation, ObservationKind,
                          Role, Species, TransparentLikelihoodModel, Visibility)
from belief_explanations import reasoning_graph


class HardConstraintsOnly:
    """Ablation: keep the same valid worlds and rules, ignore soft evidence."""

    def evaluate(self, observation, world, agents, history):
        return LikelihoodEvidence(1.0, "SOFT_EVIDENCE_DISABLED",
                                  "ハード制約のみの比較条件。", "hard-only-v1")


def decode_observation(record: dict) -> Observation:
    data = dict(record)
    data["kind"] = ObservationKind(data["kind"])
    for key, enum in (("claimed_role", Role), ("species", Species), ("visibility", Visibility)):
        if data.get(key) is not None:
            data[key] = enum(data[key])
    return Observation(**data)


def replay_record(record: dict, *, hard_only: bool = False) -> ExplainableRoleEstimator:
    """Whitelist the information accepted by the estimator boundary."""
    model = (HardConstraintsOnly() if hard_only else TransparentLikelihoodModel(
        LikelihoodParameters(**record.get("likelihood_parameters", {}))))
    estimator = ExplainableRoleEstimator(tuple(record["agents"]), record["observer"],
                                         Role(record["observer_role"]), likelihood_model=model)
    for item in record["observations"]:
        estimator.observe(decode_observation(item))
    return estimator


def evaluate_record(record: dict, *, hard_only: bool = False) -> dict:
    started = perf_counter()
    estimator = replay_record(record, hard_only=hard_only)
    elapsed_ms = (perf_counter() - started) * 1000
    # Truth is deliberately decoded only after replay, for scoring alone.
    truth = {agent: Role(role) for agent, role in record["truth"].items()}
    if set(truth) != set(estimator.agents) or Counter(truth.values()) != estimator.role_counts:
        raise ValueError("truth must be a valid complete five-player assignment")
    if truth[estimator.observer] is not estimator.observer_role:
        raise ValueError("truth disagrees with the observer's known role")
    evaluated = [agent for agent in estimator.agents if agent != estimator.observer]
    rows = []
    for agent in evaluated:
        probabilities = estimator.all_marginals()[agent]
        predicted = max(Role, key=lambda role: probabilities[role])
        rows.append({
            "correct": predicted is truth[agent],
            "confidence": probabilities[predicted],
            "log_loss": -log(max(probabilities[truth[agent]], 1e-15)),
            "brier": sum((probabilities[role] - float(role is truth[agent])) ** 2 for role in Role),
        })
    # Top-label ECE with ten equal-width bins; small demo samples are not a
    # credible calibration study. Return bin counts to make that visible.
    bins = []
    for index in range(10):
        members = [row for row in rows if min(int(row["confidence"] * 10), 9) == index]
        bins.append({"count": len(members), "gap": (
            abs(sum(row["correct"] - row["confidence"] for row in members) / len(members))
            if members else 0.0)})
    dead = {item.target for item in estimator.observations
            if item.kind in {ObservationKind.EXECUTED, ObservationKind.ATTACKED}}
    alive = record.get("alive_agents", [agent for agent in estimator.agents if agent not in dead])
    if not set(alive) <= set(estimator.agents) or set(alive) & dead:
        raise ValueError("alive_agents conflicts with observed deaths or game membership")
    decision = estimator.choose_vote(alive)
    graph = reasoning_graph(estimator, decision.target)
    world_truth = tuple(truth[agent] for agent in estimator.agents)
    top = estimator.snapshot(top_k=5).top_worlds
    maximum_constraint_error = max(
        abs(sum(estimator.marginal(agent, role) for agent in estimator.agents) - count)
        for role, count in estimator.role_counts.items())
    return {
        "label": record.get("label", "unspecified"),
        "condition": "hard-only" if hard_only else "soft-evidence",
        "sample_size": len(rows),
        "role_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "log_loss": sum(row["log_loss"] for row in rows) / len(rows),
        "brier_score": sum(row["brier"] for row in rows) / len(rows),
        "ece": sum(item["count"] * item["gap"] for item in bins) / len(rows),
        "calibration_bins": bins,
        "top5_contains_truth": any(tuple(role for _, role in item.assignments) == world_truth for item in top),
        "maximum_role_count_error": maximum_constraint_error,
        "vote_target": decision.target,
        "vote_hits_wolf": truth[decision.target] is Role.WEREWOLF,
        "uniform_random_vote_hit_expectation": sum(truth[a] is Role.WEREWOLF for a in alive
                                                   if a != estimator.observer) / (len(set(alive) - {estimator.observer})),
        "update_ms_mean": elapsed_ms / max(len(estimator.observations), 1),
        "argument": asdict(graph.argument),
        "reasoning_graph": asdict(graph),
        "snapshots": [estimator.snapshot(step).to_dict() for step in range(len(estimator.observations) + 1)],
        "limitations": "合成局面1件の動作例。勝率・汎化性能・説得効果を測定した結果ではない。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.log.read_text(encoding="utf-8-sig"))
    result = [evaluate_record(record, hard_only=mode) for mode in (True, False)]
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
