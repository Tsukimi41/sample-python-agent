"""Behavioral regression tests for the documented acceptance criteria."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from belief_model import (ExplainableRoleEstimator, InconsistentEvidenceError,
                          LikelihoodEvidence, Observation, ObservationKind,
                          Role, Species, Visibility)
from belief_evaluation import evaluate_record, replay_record
from belief_explanations import Objection, ObjectionKind, reasoning_graph, respond_to_objection

AGENTS = tuple(f"Agent[{index:02d}]" for index in range(1, 6))


def model(role=Role.VILLAGER):
    return ExplainableRoleEstimator(AGENTS, AGENTS[0], role)


def co():
    return Observation("co", ObservationKind.COMINGOUT, AGENTS[1], 1, claimed_role=Role.SEER)


def private(target=AGENTS[2]):
    return Observation("private", ObservationKind.PRIVATE_DIVINED, AGENTS[0], 1,
                       target=target, species=Species.WEREWOLF, visibility=Visibility.PRIVATE)


class CompletedRequirementsTests(unittest.TestCase):
    def test_public_view_does_not_condition_on_own_role(self):
        public = model().public_view()
        self.assertEqual(60, len(public.worlds))
        for agent in AGENTS:
            self.assertAlmostEqual(.2, public.marginal(agent, Role.WEREWOLF))

    def test_entire_public_argument_is_independent_of_private_result(self):
        first, second = model(Role.SEER), model(Role.SEER)
        first.observe(private(AGENTS[2]))
        second.observe(private(AGENTS[3]))
        for estimator in (first, second):
            estimator.observe(co())
        self.assertEqual(first.explain_suspicion(AGENTS[2]), second.explain_suspicion(AGENTS[2]))
        self.assertEqual(reasoning_graph(first, AGENTS[2]), reasoning_graph(second, AGENTS[2]))

    def test_collision_rejected_without_state_change(self):
        estimator = model()
        estimator.observe(co())
        before = estimator.snapshot()
        with self.assertRaises(ValueError):
            estimator.observe(replace(co(), actor=AGENTS[2]))
        self.assertEqual(before, estimator.snapshot())

    def test_private_information_requires_seer_role_and_owner(self):
        for estimator in (model(), model().public_view()):
            with self.assertRaises(ValueError):
                estimator.observe(private())
        with self.assertRaises(ValueError):
            model(Role.SEER).observe(replace(private(), actor=AGENTS[1]))

    def test_zero_soft_likelihood_cannot_become_a_hard_constraint(self):
        class ZeroModel:
            def evaluate(self, *args):
                return LikelihoodEvidence(0, "invalid-soft", "invalid", "test")
        estimator = ExplainableRoleEstimator(AGENTS, AGENTS[0], Role.VILLAGER,
                                            likelihood_model=ZeroModel())
        with self.assertRaises(ValueError):
            estimator.observe(co())
        self.assertEqual(0, len(estimator.observations))

    def test_all_worlds_eliminated_is_atomic_and_names_event(self):
        estimator = model(Role.SEER)
        estimator.observe(private())
        before = estimator.snapshot()
        with self.assertRaisesRegex(InconsistentEvidenceError, "contradiction"):
            estimator.observe(replace(private(), id="contradiction", species=Species.HUMAN))
        self.assertEqual(before, estimator.snapshot())

    def test_timing_changes_likelihood_and_named_trace(self):
        early, late = model(), model()
        early.observe(co())
        update = late.observe(replace(co(), turn=7))
        self.assertNotEqual(early.probabilities(), late.probabilities())
        self.assertTrue(all(item.rule_name == "LATE_SEER_COMINGOUT" for item in update.world_evidence))

    def test_self_vote_does_not_reinforce_its_own_belief(self):
        estimator = model()
        before = estimator.probabilities()
        estimator.observe(Observation("self", ObservationKind.VOTE_DECLARATION,
                                      AGENTS[0], 1, target=AGENTS[2]))
        self.assertEqual(before, estimator.probabilities())

    def test_counterfactual_replays_changed_context_without_mutation(self):
        estimator = model()
        estimator.observe(co())
        estimator.observe(replace(co(), id="counter", actor=AGENTS[2]))
        before = estimator.snapshot()
        without = estimator.without_evidence("co")
        self.assertEqual("SEER_COMINGOUT_BY_ROLE", without.updates[0].world_evidence[0].rule_name)
        self.assertEqual(before, estimator.snapshot())
        with self.assertRaises(KeyError):
            estimator.without_evidence("missing")

    def test_replay_one_hundred_times_preserves_vote_and_argument(self):
        estimator = model()
        estimator.observe(co())
        expected = estimator.choose_vote(AGENTS)
        for _ in range(100):
            self.assertEqual(expected, estimator.replay().choose_vote(AGENTS))

    def test_graph_edges_and_explanation_reference_actual_events(self):
        estimator = model()
        estimator.observe(co())
        graph = reasoning_graph(estimator, AGENTS[2])
        ids = {node.id for node in graph.nodes}
        for edge in graph.edges:
            self.assertTrue(set(edge.source_ids) <= ids)
            self.assertIn(edge.target_id, ids)
        self.assertTrue(graph.argument.evidence_ids)
        self.assertTrue(all(f"event:{event}" in ids for event in graph.argument.evidence_ids))
        self.assertEqual(len(graph.argument.evidence_ids), len(set(graph.argument.evidence_ids)))

    def test_objections_identify_premise_and_recompute_sensitivity(self):
        estimator = model()
        estimator.observe(co())
        response = respond_to_objection(estimator, AGENTS[2], Objection(ObjectionKind.OBSERVATION, "co"))
        self.assertAlmostEqual(.2, response.probability_without_evidence)
        rule = respond_to_objection(estimator, AGENTS[2], Objection(ObjectionKind.RULE, "co"))
        self.assertIn("SEER_COMINGOUT_BY_ROLE", rule.response)
        with self.assertRaises(ValueError):
            respond_to_objection(estimator, AGENTS[2], Objection(ObjectionKind.OBSERVATION, "private"))

    def test_evaluation_truth_does_not_change_replay(self):
        path = Path(__file__).resolve().parents[1] / "examples/five_player.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        first = replay_record(record).snapshot()
        altered = dict(record, truth={"server_only": "SECRET"})
        self.assertEqual(first, replay_record(altered).snapshot())
        report = evaluate_record(record)
        self.assertEqual(4, report["sample_size"])
        self.assertLess(report["maximum_role_count_error"], 1e-12)
        self.assertTrue(0 <= report["brier_score"] <= 2)
        baseline = evaluate_record(record, hard_only=True)
        self.assertAlmostEqual(.75, baseline["brier_score"])
        self.assertAlmostEqual(.25, baseline["uniform_random_vote_hit_expectation"])

    def test_team_probability_includes_possessed(self):
        estimator = model()
        self.assertAlmostEqual(.5, estimator.team_probabilities(AGENTS[1])["WOLF"])


if __name__ == "__main__":
    unittest.main()
