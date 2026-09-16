import unittest

try:
    from aiwolf import Agent, GameInfo, GameSetting, Role, Topic
    from seer import SampleSeer
    from villager import SampleVillager
except ModuleNotFoundError:
    Agent = None
    GameInfo = None
    GameSetting = None
    Role = None
    Topic = None
    SampleSeer = None
    SampleVillager = None

from belief_model import Role as BeliefRole


def game_setting():
    return GameSetting(
        {
            "enableNoAttack": False,
            "enableNoExecution": False,
            "enableRoleRequest": False,
            "maxAttackRevote": 1,
            "maxRevote": 1,
            "maxSkip": 3,
            "maxTalk": 10,
            "maxTalkTurn": 10,
            "maxWhisper": 10,
            "maxWhisperTurn": 10,
            "playerNum": 5,
            "randomSeed": 1,
            "roleNumMap": {
                "VILLAGER": 2,
                "SEER": 1,
                "POSSESSED": 1,
                "WEREWOLF": 1,
            },
            "talkOnFirstDay": True,
            "timeLimit": 1000,
            "validateUtterance": True,
            "votableInFirstDay": True,
            "voteVisible": True,
            "whisperBeforeRevote": False,
        }
    )


def game_info(talks=None, *, role="VILLAGER", divine_result=None):
    return GameInfo(
        {
            "agent": 1,
            "attackVoteList": [],
            "attackedAgent": -1,
            "cursedFox": -1,
            "day": 1,
            "divineResult": divine_result,
            "executedAgent": -1,
            "existingRoleList": ["VILLAGER", "SEER", "POSSESSED", "WEREWOLF"],
            "guardedAgent": -1,
            "lastDeadAgentList": [],
            "latestAttackVoteList": [],
            "latestExecutedAgent": -1,
            "latestVoteList": [],
            "mediumResult": None,
            "remainTalkMap": {str(i): 10 for i in range(1, 6)},
            "remainWhisperMap": {str(i): 0 for i in range(1, 6)},
            "roleMap": {"1": role},
            "statusMap": {str(i): "ALIVE" for i in range(1, 6)},
            "talkList": talks or [],
            "voteList": [],
            "whisperList": [],
        }
    )


@unittest.skipIf(Agent is None, "aiwolf package is not installed")
class SampleVillagerBeliefIntegrationTests(unittest.TestCase):
    def test_villager_uses_estimator_and_retains_vote_reason(self) -> None:
        player = SampleVillager()
        initial = game_info()
        player.initialize(initial, game_setting())
        player.day_start()
        updated = game_info(
            [
                {
                    "agent": 2,
                    "day": 1,
                    "idx": 0,
                    "text": "COMINGOUT Agent[02] SEER",
                    "turn": 0,
                },
                {
                    "agent": 2,
                    "day": 1,
                    "idx": 1,
                    "text": "DIVINED Agent[03] WEREWOLF",
                    "turn": 1,
                },
            ]
        )

        player.update(updated)
        utterance = player.talk()

        self.assertIsNotNone(player.belief_estimator)
        self.assertEqual(Agent(3), player.vote())
        self.assertEqual(Topic.VOTE, utterance.topic)
        self.assertIn("主張:", player.last_vote_explanation)
        self.assertIn("根拠:", player.last_vote_explanation)
        self.assertIn("代替仮説:", player.last_vote_explanation)
        self.assertNotIn("比較可能な別候補はいない", player.last_vote_explanation)
        self.assertIn("根拠ではなく", player.last_vote_explanation)

    def test_repeated_talk_skips_after_same_vote_has_been_declared(self) -> None:
        player = SampleVillager()
        initial = game_info()
        player.initialize(initial, game_setting())
        player.day_start()
        player.update(initial)

        first = player.talk()
        second = player.talk()

        self.assertEqual(Topic.VOTE, first.topic)
        self.assertEqual(Topic.Skip, second.topic)


@unittest.skipIf(Agent is None, "aiwolf package is not installed")
class SampleSeerBeliefIntegrationTests(unittest.TestCase):
    def test_private_result_drives_vote_without_leaking_into_public_reason(self) -> None:
        player = SampleSeer()
        info = game_info(
            role="SEER",
            divine_result={
                "agent": 1,
                "day": 1,
                "target": 4,
                "result": "WEREWOLF",
            },
        )
        player.initialize(info, game_setting())

        player.day_start()
        comingout = player.talk()
        report = player.talk()
        vote = player.talk()

        self.assertEqual(Topic.COMINGOUT, comingout.topic)
        self.assertEqual(Topic.DIVINED, report.topic)
        self.assertEqual(Topic.VOTE, vote.topic)
        self.assertEqual(Agent(4), player.vote())
        self.assertAlmostEqual(
            1.0,
            player.belief_estimator.marginal(Agent(4), BeliefRole.WEREWOLF),
        )
        self.assertNotIn("自分だけが得た", player.last_vote_explanation)
        self.assertIn("公開済みの観測だけでは", player.last_vote_explanation)


if __name__ == "__main__":
    unittest.main()
