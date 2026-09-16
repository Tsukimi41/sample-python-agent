import unittest

try:
    from aiwolf import Agent, Judge, Species as AIWolfSpecies, Talk
    from aiwolf_belief_adapter import (
        private_divination_to_observation,
        talk_to_observation,
    )
except ModuleNotFoundError:  # Core tests remain runnable without the optional SDK.
    Agent = None
    Judge = None
    AIWolfSpecies = None
    Talk = None
    private_divination_to_observation = None
    talk_to_observation = None

from belief_model import ObservationKind, Role, Species, Visibility


@unittest.skipIf(Agent is None, "aiwolf package is not installed")
class AIWolfBeliefAdapterTests(unittest.TestCase):
    def test_third_party_role_claim_is_not_self_comingout(self):
        talk = Talk(1, Agent(2), 3, "COMINGOUT Agent[03] SEER", 1)
        self.assertIsNone(talk_to_observation(talk))

    def test_translates_comingout_as_an_observation(self) -> None:
        talk = Talk(1, Agent(2), 3, "COMINGOUT Agent[02] SEER", 1)

        observation = talk_to_observation(talk)

        self.assertIsNotNone(observation)
        self.assertEqual(ObservationKind.COMINGOUT, observation.kind)
        self.assertEqual(Role.SEER, observation.claimed_role)
        self.assertEqual(Agent(2), observation.actor)

    def test_translates_divination_report_without_asserting_truth(self) -> None:
        talk = Talk(1, Agent(2), 4, "DIVINED Agent[03] WEREWOLF", 1)

        observation = talk_to_observation(talk)

        self.assertIsNotNone(observation)
        self.assertEqual(ObservationKind.DIVINED, observation.kind)
        self.assertEqual(Agent(3), observation.target)
        self.assertEqual(Species.WEREWOLF, observation.species)

    def test_ignores_unsupported_talk(self) -> None:
        talk = Talk(1, Agent(2), 5, "Skip", 1)

        self.assertIsNone(talk_to_observation(talk))

    def test_marks_own_divination_as_private_certain_evidence(self) -> None:
        judge = Judge(Agent(1), 1, Agent(4), AIWolfSpecies.WEREWOLF)

        observation = private_divination_to_observation(judge, Agent(1))

        self.assertEqual(ObservationKind.PRIVATE_DIVINED, observation.kind)
        self.assertEqual(Visibility.PRIVATE, observation.visibility)
        self.assertEqual(Species.WEREWOLF, observation.species)

    def test_rejects_another_players_private_result(self) -> None:
        judge = Judge(Agent(2), 1, Agent(4), AIWolfSpecies.WEREWOLF)

        with self.assertRaises(ValueError):
            private_divination_to_observation(judge, Agent(1))


if __name__ == "__main__":
    unittest.main()
