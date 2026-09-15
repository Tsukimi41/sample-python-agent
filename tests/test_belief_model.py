import unittest

from belief_model import (
    ExplainableRoleEstimator,
    Observation,
    ObservationKind,
    Role,
    Species,
    Visibility,
)


AGENTS = ("Agent[01]", "Agent[02]", "Agent[03]", "Agent[04]", "Agent[05]")


class ExplainableRoleEstimatorTests(unittest.TestCase):
    def make_estimator(self, role: Role = Role.VILLAGER) -> ExplainableRoleEstimator:
        return ExplainableRoleEstimator(AGENTS, AGENTS[0], role)

    def test_world_enumeration_respects_role_counts(self) -> None:
        estimator = self.make_estimator(Role.VILLAGER)

        self.assertEqual(24, len(estimator.worlds))
        for world in estimator.worlds:
            self.assertEqual(2, world.roles.count(Role.VILLAGER))
            self.assertEqual(1, world.roles.count(Role.SEER))
            self.assertEqual(1, world.roles.count(Role.POSSESSED))
            self.assertEqual(1, world.roles.count(Role.WEREWOLF))
            self.assertEqual(Role.VILLAGER, estimator.role_of(world, AGENTS[0]))

    def test_initial_marginals_are_jointly_consistent(self) -> None:
        estimator = self.make_estimator()

        for agent in AGENTS:
            self.assertAlmostEqual(
                1.0,
                sum(estimator.marginal(agent, role) for role in Role),
            )
        for role, expected_count in estimator.role_counts.items():
            self.assertAlmostEqual(
                expected_count,
                sum(estimator.marginal(agent, role) for agent in AGENTS),
            )

    def test_comingout_is_soft_evidence_not_truth(self) -> None:
        estimator = self.make_estimator()
        observation = Observation(
            id="talk:1:0",
            kind=ObservationKind.COMINGOUT,
            actor=AGENTS[1],
            claimed_role=Role.SEER,
            day=1,
            turn=0,
        )

        estimator.observe(observation)

        probability = estimator.marginal(AGENTS[1], Role.SEER)
        self.assertGreater(probability, 0.25)
        self.assertLess(probability, 1.0)

    def test_private_black_divination_is_a_hard_self_perspective_fact(self) -> None:
        estimator = self.make_estimator(Role.SEER)
        observation = Observation(
            id="private-divine:1",
            kind=ObservationKind.PRIVATE_DIVINED,
            actor=AGENTS[0],
            target=AGENTS[3],
            species=Species.WEREWOLF,
            day=1,
            visibility=Visibility.PRIVATE,
        )

        update = estimator.observe(observation)

        self.assertGreater(update.eliminated_worlds, 0)
        self.assertAlmostEqual(1.0, estimator.marginal(AGENTS[3], Role.WEREWOLF))

    def test_public_dialogue_explanation_never_leaks_private_divination(self) -> None:
        estimator = self.make_estimator(Role.SEER)
        estimator.observe(
            Observation(
                id="private-divine:1",
                kind=ObservationKind.PRIVATE_DIVINED,
                actor=AGENTS[0],
                target=AGENTS[3],
                species=Species.WEREWOLF,
                day=1,
                visibility=Visibility.PRIVATE,
            )
        )

        argument = estimator.explain_suspicion(AGENTS[3], public_only=True)
        rendered = argument.render()

        self.assertNotIn("自分だけが得た", rendered)
        self.assertIn("公開済みの観測だけでは", rendered)
        audit = estimator.audit_reasons_for(AGENTS[3])
        self.assertTrue(any(reason.visibility is Visibility.PRIVATE for reason in audit))

    def test_contradictory_public_reports_do_not_destroy_all_worlds(self) -> None:
        estimator = self.make_estimator()
        estimator.observe(
            Observation(
                id="talk:1:0",
                kind=ObservationKind.DIVINED,
                actor=AGENTS[1],
                target=AGENTS[2],
                species=Species.HUMAN,
                day=1,
                turn=0,
            )
        )
        estimator.observe(
            Observation(
                id="talk:1:1",
                kind=ObservationKind.DIVINED,
                actor=AGENTS[1],
                target=AGENTS[2],
                species=Species.WEREWOLF,
                day=1,
                turn=1,
            )
        )

        self.assertAlmostEqual(1.0, sum(estimator.probabilities()))
        self.assertTrue(any(probability > 0 for probability in estimator.probabilities()))

    def test_vote_is_deterministic_and_has_a_contestable_argument(self) -> None:
        def decide():
            estimator = self.make_estimator()
            estimator.observe(
                Observation(
                    id="vote:1:0",
                    kind=ObservationKind.VOTE,
                    actor=AGENTS[1],
                    target=AGENTS[2],
                    day=1,
                )
            )
            return estimator.choose_vote(AGENTS)

        first = decide()
        second = decide()

        self.assertEqual(first.target, second.target)
        self.assertEqual(first.argument, second.argument)
        self.assertTrue(first.argument.grounds)
        self.assertTrue(first.argument.warrants)
        self.assertTrue(first.argument.alternatives)
        self.assertIn("根拠ではなく", first.argument.confidence_summary)

    def test_observation_ids_are_idempotent(self) -> None:
        estimator = self.make_estimator()
        observation = Observation(
            id="talk:1:0",
            kind=ObservationKind.COMINGOUT,
            actor=AGENTS[1],
            claimed_role=Role.SEER,
            day=1,
        )

        first = estimator.observe(observation)
        probabilities = estimator.probabilities()
        second = estimator.observe(observation)

        self.assertIs(first, second)
        self.assertEqual(probabilities, estimator.probabilities())
        self.assertEqual(1, len(estimator.observations))


if __name__ == "__main__":
    unittest.main()
