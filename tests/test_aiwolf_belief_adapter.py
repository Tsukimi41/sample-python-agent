import unittest

try:
    from aiwolf import Agent, Talk
    from aiwolf_belief_adapter import talk_to_observation
except ModuleNotFoundError:  # Core tests remain runnable without the optional SDK.
    Agent = None
    Talk = None
    talk_to_observation = None

from belief_model import ObservationKind, Role, Species


@unittest.skipIf(Agent is None, "aiwolf package is not installed")
class AIWolfBeliefAdapterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
