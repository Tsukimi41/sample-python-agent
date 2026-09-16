"""Explainable possible-world role inference for five-player AIWolf.

The probabilities produced by this module are summaries of uncertain belief.
They are deliberately not treated as explanations.  Explanations are built
from observations, named rules, competing hypotheses, and their relations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from math import exp, isfinite, log
from typing import Hashable, Iterable, Mapping, Protocol, Sequence


AgentId = Hashable


class Role(str, Enum):
    VILLAGER = "VILLAGER"
    SEER = "SEER"
    POSSESSED = "POSSESSED"
    WEREWOLF = "WEREWOLF"


class Species(str, Enum):
    HUMAN = "HUMAN"
    WEREWOLF = "WEREWOLF"


class ObservationKind(str, Enum):
    COMINGOUT = "COMINGOUT"
    DIVINED = "DIVINED"
    VOTE_DECLARATION = "VOTE_DECLARATION"
    VOTE = "VOTE"
    EXECUTED = "EXECUTED"
    ATTACKED = "ATTACKED"
    PRIVATE_DIVINED = "PRIVATE_DIVINED"


class Visibility(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"


FIVE_PLAYER_ROLE_COUNTS: Mapping[Role, int] = {
    Role.VILLAGER: 2,
    Role.SEER: 1,
    Role.POSSESSED: 1,
    Role.WEREWOLF: 1,
}


@dataclass(frozen=True)
class Observation:
    """An observed event, kept separate from the truth it may claim."""

    id: str
    kind: ObservationKind
    actor: AgentId
    day: int
    turn: int = 0
    target: AgentId | None = None
    claimed_role: Role | None = None
    species: Species | None = None
    visibility: Visibility = Visibility.PUBLIC
    phase: str = "TALK"
    talk_index: int = 0
    source: str = "protocol"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("observation id must not be empty")
        if self.day < 0 or self.turn < 0 or self.talk_index < 0:
            raise ValueError("day and turn must be non-negative")
        if self.kind is ObservationKind.COMINGOUT and self.claimed_role is None:
            raise ValueError("COMINGOUT requires claimed_role")
        if self.kind in {
            ObservationKind.DIVINED,
            ObservationKind.PRIVATE_DIVINED,
        } and (self.target is None or self.species is None):
            raise ValueError(f"{self.kind.value} requires target and species")
        if self.kind in {
            ObservationKind.VOTE_DECLARATION,
            ObservationKind.VOTE,
            ObservationKind.EXECUTED,
            ObservationKind.ATTACKED,
        } and self.target is None:
            raise ValueError(f"{self.kind.value} requires target")
        if (
            self.kind is ObservationKind.PRIVATE_DIVINED
            and self.visibility is not Visibility.PRIVATE
        ):
            raise ValueError("PRIVATE_DIVINED must have PRIVATE visibility")


@dataclass(frozen=True)
class World:
    """A complete role assignment in the estimator's fixed agent order."""

    roles: tuple[Role, ...]


@dataclass(frozen=True)
class LikelihoodEvidence:
    likelihood: float
    rule_name: str
    warrant: str
    model_version: str


@dataclass(frozen=True)
class LikelihoodParameters:
    """Auditable baseline parameters; each value is an explicit hypothesis."""

    version: str = "transparent-v2"
    seer_co_if_seer: float = 0.82
    seer_co_if_possessed: float = 0.65
    seer_co_if_werewolf: float = 0.30
    seer_co_if_villager: float = 0.05
    other_co_if_truthful: float = 0.75
    other_co_if_deceptive: float = 0.12
    divined_match_if_seer: float = 0.95
    divined_mismatch_if_seer: float = 0.05
    divined_match_if_werewolf: float = 0.75
    divined_mismatch_if_werewolf: float = 0.25
    divined_if_possessed: float = 0.50
    divined_if_villager: float = 0.15
    contradiction_if_seer: float = 0.03
    contradiction_if_villager: float = 0.08
    contradiction_if_possessed: float = 0.20
    contradiction_if_werewolf: float = 0.20
    actual_vote_village_to_wolf: float = 0.55
    actual_vote_village_to_other: float = 0.15
    actual_vote_wolf_side_to_wolf: float = 0.10
    actual_vote_wolf_side_to_other: float = 0.30
    declared_vote_village_to_wolf: float = 0.40
    declared_vote_village_to_other: float = 0.20
    declared_vote_wolf_side_to_wolf: float = 0.15
    declared_vote_wolf_side_to_other: float = 0.28
    late_co_if_seer: float = 0.70
    late_co_if_other: float = 0.95
    changed_vote_if_village: float = 0.65
    changed_vote_if_wolf_side: float = 0.85

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("likelihood parameter version must not be empty")
        for name, value in self.__dict__.items():
            if name == "version":
                continue
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in the interval (0, 1]")


class LikelihoodModel(Protocol):
    def evaluate(
        self,
        observation: Observation,
        world: World,
        agents: Sequence[AgentId],
        history: Sequence[Observation],
    ) -> LikelihoodEvidence:
        """Return P(observation | world, context) and its named rationale."""


@dataclass(frozen=True)
class Reason:
    observation_id: str
    ground: str
    warrant: str
    hypothesis: str
    direction: str
    visibility: Visibility
    probability_change: float
    rule_name: str
    model_version: str


@dataclass(frozen=True)
class Argument:
    """A contestable explanation, suitable for later dialogue generation."""

    claim: str
    grounds: tuple[str, ...]
    warrants: tuple[str, ...]
    alternatives: tuple[str, ...]
    counterarguments: tuple[str, ...]
    confidence_summary: str
    audience: AgentId | None = None
    evidence_ids: tuple[str, ...] = ()
    rebuttals: tuple[str, ...] = ()
    disclosure_policy: str = "公開履歴だけで再計算し、自己役職・私有観測を開示しない。"

    def render(self) -> str:
        parts = [f"主張: {self.claim}"]
        parts.extend(f"根拠: {ground}" for ground in self.grounds)
        parts.extend(f"理由: {warrant}" for warrant in self.warrants)
        parts.extend(f"代替仮説: {alternative}" for alternative in self.alternatives)
        parts.extend(f"想定反論: {item}" for item in self.counterarguments)
        parts.extend(f"応答: {item}" for item in self.rebuttals)
        parts.append(f"参考値: {self.confidence_summary}")
        return "\n".join(parts)


@dataclass(frozen=True)
class BeliefUpdate:
    observation: Observation
    eliminated_worlds: int
    reasons: tuple[Reason, ...]
    world_evidence: tuple[LikelihoodEvidence, ...] = ()


@dataclass(frozen=True)
class VoteDecision:
    target: AgentId
    score: float
    tie_break_rule: str
    argument: Argument
    scores: tuple[tuple[AgentId, float], ...] = ()


@dataclass(frozen=True)
class WorldSummary:
    assignments: tuple[tuple[AgentId, Role], ...]
    probability: float


@dataclass(frozen=True)
class BeliefSnapshot:
    """Immutable, inspectable belief state after a particular event."""

    step: int
    observation_id: str | None
    marginals: tuple[tuple[AgentId, tuple[tuple[Role, float], ...]], ...]
    top_worlds: tuple[WorldSummary, ...]

    def probability(self, agent: AgentId, role: Role) -> float:
        for snapshot_agent, role_probabilities in self.marginals:
            if snapshot_agent == agent:
                return dict(role_probabilities)[role]
        raise KeyError(agent)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible view for evaluation and presentations."""

        return {
            "step": self.step,
            "observation_id": self.observation_id,
            "marginals": {
                str(agent): {
                    role.value: probability
                    for role, probability in role_probabilities
                }
                for agent, role_probabilities in self.marginals
            },
            "top_worlds": [
                {
                    "assignments": {
                        str(agent): role.value
                        for agent, role in world.assignments
                    },
                    "probability": world.probability,
                }
                for world in self.top_worlds
            ],
        }


class InconsistentEvidenceError(RuntimeError):
    """Raised instead of silently resetting when no possible world remains."""


class TransparentLikelihoodModel:
    """Small, explicit baseline likelihoods for the first implementation.

    These values are hypotheses, not learned truths.  Every value is named and
    replaceable so later log-based or neural likelihood models can implement
    the same interface.
    """

    def __init__(self, parameters: LikelihoodParameters | None = None) -> None:
        self.parameters = parameters or LikelihoodParameters()

    def _evidence(
        self, likelihood: float, rule_name: str, warrant: str
    ) -> LikelihoodEvidence:
        return LikelihoodEvidence(
            likelihood,
            rule_name,
            warrant,
            self.parameters.version,
        )

    def evaluate(
        self,
        observation: Observation,
        world: World,
        agents: Sequence[AgentId],
        history: Sequence[Observation],
    ) -> LikelihoodEvidence:
        role_by_agent = dict(zip(agents, world.roles))
        actor_role = role_by_agent[observation.actor]
        parameters = self.parameters

        if observation.kind is ObservationKind.COMINGOUT:
            if observation.claimed_role is Role.SEER:
                likelihood = {
                    Role.SEER: parameters.seer_co_if_seer,
                    Role.POSSESSED: parameters.seer_co_if_possessed,
                    Role.WEREWOLF: parameters.seer_co_if_werewolf,
                    Role.VILLAGER: parameters.seer_co_if_villager,
                }[actor_role]
                # A late/counter claim is a different observation context.
                late = observation.day > 1 or observation.turn >= 5 or any(
                    item.kind is ObservationKind.COMINGOUT
                    and item.claimed_role is Role.SEER
                    and item.actor != observation.actor for item in history
                )
                if late:
                    likelihood *= (parameters.late_co_if_seer if actor_role is Role.SEER
                                   else parameters.late_co_if_other)
                return self._evidence(
                    likelihood,
                    "LATE_SEER_COMINGOUT" if late else "SEER_COMINGOUT_BY_ROLE",
                    "占い師COの起こりやすさは役職で異なる。" + (
                        "遅い時点または対抗CO後の宣言として、時点の尤度係数を適用する。"
                        if late else "初期のCOとして比較する。"),
                )
            likelihood = (
                parameters.other_co_if_truthful
                if actor_role is observation.claimed_role
                else parameters.other_co_if_deceptive
            )
            return self._evidence(
                likelihood,
                "ROLE_COMINGOUT_BY_ROLE",
                "自分の役職をCOする場合と、別の役職を騙る場合を同一視しない。",
            )

        if observation.kind is ObservationKind.DIVINED:
            assert observation.target is not None
            assert observation.species is not None
            target_is_wolf = role_by_agent[observation.target] is Role.WEREWOLF
            report_is_wolf = observation.species is Species.WEREWOLF
            report_matches_world = target_is_wolf == report_is_wolf

            if actor_role is Role.SEER:
                likelihood = (
                    parameters.divined_match_if_seer
                    if report_matches_world
                    else parameters.divined_mismatch_if_seer
                )
            elif actor_role is Role.WEREWOLF:
                likelihood = (
                    parameters.divined_match_if_werewolf
                    if report_matches_world
                    else parameters.divined_mismatch_if_werewolf
                )
            elif actor_role is Role.POSSESSED:
                likelihood = parameters.divined_if_possessed
            else:
                likelihood = parameters.divined_if_villager

            warrant = (
                "占い報告と役職世界の整合性を比較する。真占い師は一致しやすく、"
                "人狼は正確に真似られ、狂人は偶然一致し得るため、報告だけでは確定しない。"
            )

            contradictions = [
                previous
                for previous in history
                if previous.kind is ObservationKind.DIVINED
                and previous.actor == observation.actor
                and previous.target == observation.target
                and previous.species is not observation.species
            ]
            if contradictions:
                contradiction_factor = {
                    Role.SEER: parameters.contradiction_if_seer,
                    Role.VILLAGER: parameters.contradiction_if_villager,
                    Role.POSSESSED: parameters.contradiction_if_possessed,
                    Role.WEREWOLF: parameters.contradiction_if_werewolf,
                }[actor_role]
                likelihood *= contradiction_factor
                warrant += " 同じ対象への矛盾報告は、特に真占い師仮説を弱める。"

            return self._evidence(
                likelihood,
                "DIVINATION_REPORT_COMPATIBILITY",
                warrant,
            )

        if observation.kind in {
            ObservationKind.VOTE_DECLARATION,
            ObservationKind.VOTE,
        }:
            assert observation.target is not None
            target_is_wolf = role_by_agent[observation.target] is Role.WEREWOLF
            village_side = actor_role in {Role.VILLAGER, Role.SEER}
            if observation.kind is ObservationKind.VOTE:
                likelihood = (
                    parameters.actual_vote_village_to_wolf
                    if village_side and target_is_wolf
                    else parameters.actual_vote_village_to_other
                    if village_side
                    else parameters.actual_vote_wolf_side_to_wolf
                    if target_is_wolf
                    else parameters.actual_vote_wolf_side_to_other
                )
                rule_name = "ACTUAL_VOTE_BY_TEAM"
                warrant = "実投票は、村側なら人狼へ、人狼陣営なら人間へ向きやすいという弱い証拠として扱う。"
                declarations = [item for item in history
                                if item.kind is ObservationKind.VOTE_DECLARATION
                                and item.actor == observation.actor and item.day == observation.day]
                if declarations and declarations[-1].target != observation.target:
                    likelihood *= (parameters.changed_vote_if_village if village_side
                                   else parameters.changed_vote_if_wolf_side)
                    rule_name = "VOTE_CHANGED_FROM_DECLARATION"
                    warrant += " 最後の投票宣言からの変更を、追加のソフト証拠として評価する。"
            else:
                likelihood = (
                    parameters.declared_vote_village_to_wolf
                    if village_side and target_is_wolf
                    else parameters.declared_vote_village_to_other
                    if village_side
                    else parameters.declared_vote_wolf_side_to_wolf
                    if target_is_wolf
                    else parameters.declared_vote_wolf_side_to_other
                )
                rule_name = "DECLARED_VOTE_BY_TEAM"
                warrant = "投票宣言は戦略的に偽れるため、実投票より弱い証拠として扱う。"
            return self._evidence(likelihood, rule_name, warrant)

        return self._evidence(
            1.0,
            "NEUTRAL_PUBLIC_EVENT",
            "この公開イベントは、現在の基準モデルでは役職の相対重みを変えない。",
        )


class ExplainableRoleEstimator:
    """Exact possible-world inference for the five-player role set."""

    def __init__(
        self,
        agents: Sequence[AgentId],
        observer: AgentId | None,
        observer_role: Role | None,
        *,
        role_counts: Mapping[Role, int] = FIVE_PLAYER_ROLE_COUNTS,
        likelihood_model: LikelihoodModel | None = None,
    ) -> None:
        if len(set(agents)) != len(agents):
            raise ValueError("agents must be unique")
        if observer is not None and observer not in agents:
            raise ValueError("observer must be one of agents")
        if (observer is None) != (observer_role is None):
            raise ValueError("public perspective requires both observer and role to be None")
        if len(agents) != 5 or dict(role_counts) != dict(FIVE_PLAYER_ROLE_COUNTS):
            raise ValueError("exact inference supports only the documented five-player composition")
        if sum(role_counts.values()) != len(agents):
            raise ValueError("role counts must equal number of agents")
        if observer_role is not None and role_counts.get(observer_role, 0) < 1:
            raise ValueError("observer role is absent from role counts")

        self.agents = tuple(agents)
        self.observer = observer
        self.observer_role = observer_role
        self.role_counts = dict(role_counts)
        self.likelihood_model = likelihood_model or TransparentLikelihoodModel()
        self._agent_index = {agent: index for index, agent in enumerate(self.agents)}
        self.worlds = self._generate_worlds()
        uniform_log_weight = -log(len(self.worlds))
        self._log_weights = [uniform_log_weight for _ in self.worlds]
        self.observations: list[Observation] = []
        self.updates: list[BeliefUpdate] = []
        self._seen_observations: dict[str, BeliefUpdate] = {}
        self._marginal_snapshots = [self.all_marginals()]
        self._log_weight_snapshots = [tuple(self._log_weights)]

    def _generate_worlds(self) -> tuple[World, ...]:
        remaining_counts = dict(self.role_counts)
        if self.observer_role is not None:
            remaining_counts[self.observer_role] -= 1
        remaining_roles: list[Role] = []
        for role, count in remaining_counts.items():
            remaining_roles.extend([role] * count)

        others = [agent for agent in self.agents if agent != self.observer]
        unique_assignments = sorted(
            set(permutations(remaining_roles)),
            key=lambda roles: tuple(role.value for role in roles),
        )
        worlds = []
        for assignment in unique_assignments:
            by_agent = dict(zip(others, assignment))
            if self.observer is not None:
                by_agent[self.observer] = self.observer_role
            worlds.append(World(tuple(by_agent[agent] for agent in self.agents)))
        if not worlds:
            raise ValueError("no valid role assignments")
        return tuple(worlds)

    def role_of(self, world: World, agent: AgentId) -> Role:
        return world.roles[self._agent_index[agent]]

    def probabilities(self) -> tuple[float, ...]:
        return tuple(exp(value) for value in self._log_weights)

    def marginal(self, agent: AgentId, role: Role) -> float:
        index = self._agent_index[agent]
        return sum(
            probability
            for world, probability in zip(self.worlds, self.probabilities())
            if world.roles[index] is role
        )

    def all_marginals(self) -> dict[AgentId, dict[Role, float]]:
        return {
            agent: {role: self.marginal(agent, role) for role in self.role_counts}
            for agent in self.agents
        }

    def observe(self, observation: Observation) -> BeliefUpdate:
        """Apply one observation exactly once and retain its reasoning trace."""

        if observation.id in self._seen_observations:
            previous = self._seen_observations[observation.id]
            if previous.observation != observation:
                raise ValueError(f"conflicting content for observation id {observation.id!r}")
            return previous
        if observation.actor not in self._agent_index:
            raise ValueError("observation actor is not in this game")
        if observation.target is not None and observation.target not in self._agent_index:
            raise ValueError("observation target is not in this game")
        if (
            observation.visibility is Visibility.PRIVATE
            and observation.actor != self.observer
        ):
            raise ValueError("cannot import another player's private observation")
        if (observation.kind is ObservationKind.PRIVATE_DIVINED
                and self.observer_role is not Role.SEER):
            raise ValueError("only a seer may receive private divination")

        before = self.all_marginals()
        new_log_weights: list[float] = []
        evidence_by_world: list[LikelihoodEvidence] = []
        eliminated_worlds = 0

        for world, old_log_weight in zip(self.worlds, self._log_weights):
            hard_allowed, hard_evidence = self._hard_constraint(observation, world)
            evidence = hard_evidence or self.likelihood_model.evaluate(
                observation, world, self.agents, self.observations
            )
            # Our own policy outputs are interventions, not independent evidence
            # about hidden roles. A public observer can still interpret them.
            if hard_evidence is None and observation.actor == self.observer:
                evidence = LikelihoodEvidence(1.0, "SELF_ACTION_NOT_EVIDENCE",
                    "自分の発言・投票から自分の信念を増幅しない。", "game-rules-v1")
            evidence_by_world.append(evidence)
            if not hard_allowed:
                new_log_weights.append(float("-inf"))
                if isfinite(old_log_weight):
                    eliminated_worlds += 1
                continue
            if not 0 < evidence.likelihood <= 1 or not isfinite(evidence.likelihood):
                raise ValueError(
                    f"invalid likelihood {evidence.likelihood} from {evidence.rule_name}"
                )
            new_log_weights.append(old_log_weight + log(evidence.likelihood))

        self._log_weights = self._normalize(new_log_weights, observation)
        after = self.all_marginals()
        reasons = self._make_reasons(observation, before, after, evidence_by_world)
        update = BeliefUpdate(observation, eliminated_worlds, reasons, tuple(evidence_by_world))
        self.observations.append(observation)
        self.updates.append(update)
        self._seen_observations[observation.id] = update
        self._marginal_snapshots.append(after)
        self._log_weight_snapshots.append(tuple(self._log_weights))
        return update

    def snapshot(self, step: int | None = None, *, top_k: int = 5) -> BeliefSnapshot:
        """Return the initial or post-observation state without mutating it.

        Step 0 is the prior.  Step n is the state after observation n.  Keeping
        this convention explicit prevents explanations from accidentally using
        evidence that had not yet occurred at the requested point in time.
        """

        selected_step = len(self.observations) if step is None else step
        if not 0 <= selected_step <= len(self.observations):
            raise IndexError("snapshot step is outside the belief history")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        marginals = self._marginal_snapshots[selected_step]
        frozen_marginals = tuple(
            (
                agent,
                tuple((role, marginals[agent][role]) for role in self.role_counts),
            )
            for agent in self.agents
        )
        log_weights = self._log_weight_snapshots[selected_step]
        ranked_worlds = sorted(
            zip(self.worlds, log_weights),
            key=lambda item: -item[1],
        )[:top_k]
        top_worlds = tuple(
            WorldSummary(
                assignments=tuple(zip(self.agents, world.roles)),
                probability=exp(log_weight),
            )
            for world, log_weight in ranked_worlds
            if isfinite(log_weight)
        )
        observation_id = (
            None if selected_step == 0 else self.observations[selected_step - 1].id
        )
        return BeliefSnapshot(
            step=selected_step,
            observation_id=observation_id,
            marginals=frozen_marginals,
            top_worlds=top_worlds,
        )

    def timeline(
        self, agent: AgentId, role: Role = Role.WEREWOLF
    ) -> tuple[tuple[str | None, float], ...]:
        """Return the belief trajectory for one explicit role hypothesis."""

        if agent not in self._agent_index:
            raise KeyError(agent)
        return tuple(
            (
                None if step == 0 else self.observations[step - 1].id,
                snapshot[agent][role],
            )
            for step, snapshot in enumerate(self._marginal_snapshots)
        )

    def replay(self) -> ExplainableRoleEstimator:
        """Rebuild belief from the event log to verify deterministic replay."""

        replayed = ExplainableRoleEstimator(
            self.agents,
            self.observer,
            self.observer_role,
            role_counts=self.role_counts,
            likelihood_model=self.likelihood_model,
        )
        for observation in self.observations:
            replayed.observe(observation)
        return replayed

    def _hard_constraint(
        self, observation: Observation, world: World
    ) -> tuple[bool, LikelihoodEvidence | None]:
        if observation.kind is ObservationKind.PRIVATE_DIVINED:
            assert observation.target is not None
            assert observation.species is not None
            target_is_wolf = self.role_of(world, observation.target) is Role.WEREWOLF
            observed_wolf = observation.species is Species.WEREWOLF
            allowed = target_is_wolf == observed_wolf
            return allowed, LikelihoodEvidence(
                1.0 if allowed else 0.0,
                "PRIVATE_DIVINATION_TRUTH",
                "自分が得た占い結果は自己視点では確定情報であり、矛盾する役職世界を除外する。",
                "game-rules-v1",
            )
        if observation.kind is ObservationKind.ATTACKED:
            assert observation.target is not None
            allowed = self.role_of(world, observation.target) is not Role.WEREWOLF
            return allowed, LikelihoodEvidence(
                1.0 if allowed else 0.0,
                "ATTACKED_PLAYER_IS_NOT_WEREWOLF",
                "襲撃され死亡したプレイヤーは、人狼自身ではないというゲーム規則を適用する。",
                "game-rules-v1",
            )
        return True, None

    @staticmethod
    def _normalize(
        log_weights: Sequence[float], observation: Observation
    ) -> list[float]:
        finite_weights = [value for value in log_weights if isfinite(value)]
        if not finite_weights:
            raise InconsistentEvidenceError(
                f"observation {observation.id!r} eliminated every possible world"
            )
        maximum = max(finite_weights)
        log_total = maximum + log(
            sum(exp(value - maximum) for value in finite_weights)
        )
        return [
            value - log_total if isfinite(value) else float("-inf")
            for value in log_weights
        ]

    def _make_reasons(
        self,
        observation: Observation,
        before: Mapping[AgentId, Mapping[Role, float]],
        after: Mapping[AgentId, Mapping[Role, float]],
        evidence_by_world: Sequence[LikelihoodEvidence],
    ) -> tuple[Reason, ...]:
        representative = evidence_by_world[0]
        reasons: list[Reason] = []
        for agent in self.agents:
            change = after[agent][Role.WEREWOLF] - before[agent][Role.WEREWOLF]
            if abs(change) < 1e-12:
                continue
            direction = "supports" if change > 0 else "weakens"
            reasons.append(
                Reason(
                    observation_id=observation.id,
                    ground=self._describe_observation(observation),
                    warrant=representative.warrant,
                    hypothesis=f"{agent}が{Role.WEREWOLF.value}である",
                    direction=direction,
                    visibility=observation.visibility,
                    probability_change=change,
                    rule_name=representative.rule_name,
                    model_version=representative.model_version,
                )
            )
        return tuple(reasons)

    @staticmethod
    def _describe_observation(observation: Observation) -> str:
        prefix = f"{observation.day}日目・第{observation.turn}発言"
        if observation.kind is ObservationKind.COMINGOUT:
            return (
                f"{prefix}に{observation.actor}が"
                f"{observation.claimed_role.value}をCOした。"
            )
        if observation.kind in {
            ObservationKind.DIVINED,
            ObservationKind.PRIVATE_DIVINED,
        }:
            scope = "自分だけが得た" if observation.visibility is Visibility.PRIVATE else "公開された"
            return (
                f"{prefix}に{scope}情報として、{observation.actor}が"
                f"{observation.target}を{observation.species.value}と判定した。"
            )
        if observation.kind is ObservationKind.VOTE_DECLARATION:
            return f"{prefix}に{observation.actor}が{observation.target}への投票を宣言した。"
        if observation.kind is ObservationKind.VOTE:
            return f"{observation.day}日目に{observation.actor}が{observation.target}へ実投票した。"
        if observation.kind is ObservationKind.ATTACKED:
            return f"{observation.day}日目に{observation.target}が襲撃された。"
        if observation.kind is ObservationKind.EXECUTED:
            return f"{observation.day}日目に{observation.target}が処刑された。"
        return f"{observation.kind.value}: {observation.actor}"

    def choose_vote(
        self,
        alive_agents: Iterable[AgentId],
        *,
        eligible_candidates: Iterable[AgentId] | None = None,
        audience: AgentId | None = None,
    ) -> VoteDecision:
        alive_set = set(alive_agents)
        eligible_set = (
            alive_set if eligible_candidates is None else set(eligible_candidates)
        )
        candidates = [
            agent
            for agent in self.agents
            if agent in alive_set
            and agent in eligible_set
            and agent != self.observer
        ]
        if not candidates:
            raise ValueError("there is no living vote candidate")
        scores = {agent: self.marginal(agent, Role.WEREWOLF) for agent in candidates}
        target = max(candidates, key=lambda agent: (scores[agent], -self._agent_index[agent]))
        return VoteDecision(
            target=target,
            score=scores[target],
            tie_break_rule="同点の場合はゲーム開始時のエージェント順を優先する。",
            argument=self.explain_suspicion(
                target,
                [
                    agent
                    for agent in self.agents
                    if agent in alive_set and agent != self.observer
                ],
                audience=audience,
            ),
            scores=tuple(scores.items()),
        )

    def explain_suspicion(
        self,
        target: AgentId,
        candidates: Iterable[AgentId] | None = None,
        *,
        audience: AgentId | None = None,
        public_only: bool = True,
    ) -> Argument:
        if target not in self._agent_index:
            raise ValueError("target is not in this game")
        # Filtering private sentences is insufficient: probabilities, alternative
        # ordering and public-event deltas can also reveal the private result.
        if public_only and self.observer is not None:
            return self.public_view().explain_suspicion(
                target, candidates, audience=audience, public_only=True)
        candidate_list = [
            agent
            for agent in (candidates or self.agents)
            if agent != self.observer and agent != target
        ]

        target_reasons: list[Reason] = []
        for update_index, update in enumerate(self.updates):
            for reason in update.reasons:
                if reason.hypothesis != f"{target}が{Role.WEREWOLF.value}である":
                    continue
                if public_only and reason.visibility is Visibility.PRIVATE:
                    continue
                if reason.direction != "supports":
                    continue
                change = (
                    self._marginal_snapshots[update_index + 1][target][Role.WEREWOLF]
                    - self._marginal_snapshots[update_index][target][Role.WEREWOLF]
                )
                if change > 1e-12:
                    target_reasons.append(
                        Reason(
                            reason.observation_id,
                            reason.ground,
                            reason.warrant,
                            f"{target}が{Role.WEREWOLF.value}である",
                            reason.direction,
                            reason.visibility,
                            change,
                            reason.rule_name,
                            reason.model_version,
                        )
                    )

        target_reasons.sort(key=lambda reason: -abs(reason.probability_change))
        selected = target_reasons[:3]
        if selected:
            grounds = tuple(dict.fromkeys(reason.ground for reason in selected))
            warrants = tuple(dict.fromkeys(reason.warrant for reason in selected))
        else:
            grounds = ("公開済みの観測だけでは、この候補を強く支持する個別根拠がまだない。",)
            warrants = (
                "根拠が不足している場合は、確率値を理由の代用品にせず、判断が暫定的だと明示する。",
            )

        ranked_alternatives = sorted(
            [agent for agent in candidate_list if self.marginal(agent, Role.WEREWOLF) > 0],
            key=lambda agent: (-self.marginal(agent, Role.WEREWOLF), self._agent_index[agent]),
        )
        alternatives: tuple[str, ...]
        if ranked_alternatives:
            alternative = ranked_alternatives[0]
            alternatives = (
                f"{alternative}が人狼である世界も残る。" + (
                    "この別仮説は現在の証拠モデルで対象仮説以上に支持され、公開根拠だけでは対象を優先できない。"
                    if self.marginal(alternative, Role.WEREWOLF) >= self.marginal(target, Role.WEREWOLF)
                    else "観測と各役職の行動尤度を統合すると対象仮説が優勢だが、別仮説を排除はできない。"),
            )
        else:
            alternatives = ("指定候補の範囲では、正の重みを持つ別の人狼仮説はない。",)

        opposing_reasons = [reason for reason in self.audit_reasons_for(target)
                            if reason.direction == "weakens"
                            and (not public_only or reason.visibility is Visibility.PUBLIC)]
        return Argument(
            claim=f"{target}を投票候補として検討する。公開根拠による支持と留保を示す。" if public_only
                  else f"現時点では{target}を投票候補とする。",
            grounds=grounds,
            warrants=warrants,
            alternatives=alternatives,
            counterarguments=(
                "発言は戦略、ブラフ、推理途中の誤りでも説明できるため、単独の発言だけでは役職を確定できない。",
            ) + tuple(reason.ground + " この観測の更新は対象の人狼仮説を弱めた。"
                      for reason in opposing_reasons[-2:]),
            confidence_summary=(
                f"統合後の{target}人狼確率は"
                f"{self.marginal(target, Role.WEREWOLF):.1%}。これは根拠ではなく不確実性の要約である。"
            ),
            audience=audience,
            evidence_ids=tuple(reason.observation_id for reason in selected),
        )

    def public_view(self) -> ExplainableRoleEstimator:
        """Recompute all 60 worlds without conditioning on any player's role."""
        public = ExplainableRoleEstimator(self.agents, None, None,
                                          likelihood_model=self.likelihood_model)
        for observation in self.observations:
            if observation.visibility is Visibility.PUBLIC:
                public.observe(observation)
        return public

    def without_evidence(self, observation_id: str) -> ExplainableRoleEstimator:
        """Leave-one-event-out replay, including changed downstream context.

        This is sensitivity analysis, not a causal effect of actually changing
        another player's behavior. The original estimator remains unchanged.
        """
        if observation_id not in self._seen_observations:
            raise KeyError(observation_id)
        replayed = ExplainableRoleEstimator(self.agents, self.observer, self.observer_role,
                                            likelihood_model=self.likelihood_model)
        for observation in self.observations:
            if observation.id != observation_id:
                replayed.observe(observation)
        return replayed

    def team_probabilities(self, agent: AgentId) -> dict[str, float]:
        wolf_side = self.marginal(agent, Role.WEREWOLF) + self.marginal(agent, Role.POSSESSED)
        return {"VILLAGE": 1.0 - wolf_side, "WOLF": wolf_side}

    def audit_reasons_for(self, target: AgentId) -> tuple[Reason, ...]:
        """Return public and private calculation traces for internal auditing."""

        reasons: list[Reason] = []
        for index, update in enumerate(self.updates):
            change = (
                self._marginal_snapshots[index + 1][target][Role.WEREWOLF]
                - self._marginal_snapshots[index][target][Role.WEREWOLF]
            )
            if abs(change) < 1e-12:
                continue
            template = update.reasons[0] if update.reasons else None
            if template is None:
                continue
            reasons.append(
                Reason(
                    observation_id=update.observation.id,
                    ground=self._describe_observation(update.observation),
                    warrant=template.warrant,
                    hypothesis=f"{target}が{Role.WEREWOLF.value}である",
                    direction="supports" if change > 0 else "weakens",
                    visibility=update.observation.visibility,
                    probability_change=change,
                    rule_name=template.rule_name,
                    model_version=template.model_version,
                )
            )
        return tuple(reasons)
