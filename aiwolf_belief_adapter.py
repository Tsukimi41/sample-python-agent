"""Boundary adapter between the aiwolf package and the belief model.

Keeping this translation outside the inference engine prevents accidental use
of server-only information and keeps the core independently testable.
"""

from __future__ import annotations

from aiwolf import Content, GameInfo, GameSetting, Judge, Talk, Topic

from belief_model import (
    ExplainableRoleEstimator,
    Observation,
    ObservationKind,
    Role,
    Species,
    Visibility,
)


SUPPORTED_ROLES = frozenset(role.value for role in Role)
SUPPORTED_SPECIES = frozenset(species.value for species in Species)


def create_five_player_estimator(
    game_info: GameInfo, game_setting: GameSetting
) -> ExplainableRoleEstimator | None:
    """Create the MVP estimator only when its role assumptions are satisfied."""

    role_counts = {
        role.value: count
        for role, count in game_setting.role_num_map.items()
        if count > 0
    }
    if len(game_info.agent_list) != 5:
        return None
    if set(role_counts) != SUPPORTED_ROLES:
        return None
    if role_counts != {
        Role.VILLAGER.value: 2,
        Role.SEER.value: 1,
        Role.POSSESSED.value: 1,
        Role.WEREWOLF.value: 1,
    }:
        return None
    if game_info.my_role.value not in SUPPORTED_ROLES:
        return None
    return ExplainableRoleEstimator(
        game_info.agent_list,
        game_info.me,
        Role(game_info.my_role.value),
    )


def talk_to_observation(talk: Talk) -> Observation | None:
    """Translate supported protocol utterances without treating them as truth."""

    content = Content.compile(talk.text)
    common = {
        "id": f"talk:{talk.day}:{talk.idx}",
        "actor": talk.agent,
        "day": talk.day,
        "turn": talk.turn,
        "talk_index": talk.idx,
    }
    if content.topic is Topic.COMINGOUT and content.role.value in SUPPORTED_ROLES:
        if content.target != talk.agent:
            return None  # A claim about someone else is not a self role claim.
        return Observation(
            kind=ObservationKind.COMINGOUT,
            claimed_role=Role(content.role.value),
            **common,
        )
    if content.topic is Topic.DIVINED and content.result.value in SUPPORTED_SPECIES:
        return Observation(
            kind=ObservationKind.DIVINED,
            target=content.target,
            species=Species(content.result.value),
            **common,
        )
    if content.topic is Topic.VOTE:
        return Observation(
            kind=ObservationKind.VOTE_DECLARATION,
            target=content.target,
            **common,
        )
    return None


def private_divination_to_observation(
    judge: Judge, observer: object
) -> Observation:
    """Translate a seer's own result as private, certain evidence."""

    if judge.agent != observer:
        raise ValueError("private divination result does not belong to observer")
    if judge.result.value not in SUPPORTED_SPECIES:
        raise ValueError("unsupported divination species")
    return Observation(
        id=f"private-divined:{judge.day}:{judge.target}",
        kind=ObservationKind.PRIVATE_DIVINED,
        actor=observer,
        target=judge.target,
        species=Species(judge.result.value),
        day=judge.day,
        visibility=Visibility.PRIVATE,
        phase="NIGHT",
        source="own_divination",
    )


def observe_new_game_info(
    estimator: ExplainableRoleEstimator,
    game_info: GameInfo,
    *,
    talk_start_index: int,
) -> None:
    """Import only observations visible to the receiving player."""

    for talk in game_info.talk_list[talk_start_index:]:
        observation = talk_to_observation(talk)
        if observation is not None:
            estimator.observe(observation)

    for index, vote in enumerate(game_info.vote_list):
        estimator.observe(
            Observation(
                id=f"vote:{vote.day}:{index}:{vote.agent}:{vote.target}",
                kind=ObservationKind.VOTE,
                actor=vote.agent,
                target=vote.target,
                day=vote.day,
                phase="VOTE",
            )
        )

    if game_info.executed_agent is not None:
        estimator.observe(
            Observation(
                id=f"executed:{game_info.day}:{game_info.executed_agent}",
                kind=ObservationKind.EXECUTED,
                actor=game_info.executed_agent,
                target=game_info.executed_agent,
                day=game_info.day,
            )
        )

    # attacked_agent can denote the attack target even if it survived.
    # Only an observed death justifies the non-wolf hard constraint.
    if (game_info.attacked_agent is not None
            and game_info.attacked_agent in game_info.last_dead_agent_list):
        estimator.observe(
            Observation(
                id=f"attacked:{game_info.day}:{game_info.attacked_agent}",
                kind=ObservationKind.ATTACKED,
                actor=game_info.attacked_agent,
                target=game_info.attacked_agent,
                day=game_info.day,
            )
        )
