"""Trace-backed argument graphs and structured objections.

These are inspectable data, not generated persuasion. Public graphs are built
from a separate public posterior, so even their numeric deltas are shareable.
"""

from dataclasses import dataclass
from enum import Enum

from belief_model import AgentId, Argument, ExplainableRoleEstimator, Observation, Role


@dataclass(frozen=True)
class ReasoningNode:
    id: str
    kind: str
    content: str


@dataclass(frozen=True)
class ReasoningEdge:
    source_ids: tuple[str, ...]
    target_id: str
    relation: str


@dataclass(frozen=True)
class ReasoningGraph:
    nodes: tuple[ReasoningNode, ...]
    edges: tuple[ReasoningEdge, ...]
    argument: Argument


def reasoning_graph(estimator: ExplainableRoleEstimator, target: AgentId,
                    *, public_only: bool = True) -> ReasoningGraph:
    """Connect each actual update to the hypothesis it supports or weakens."""
    view = estimator.public_view() if public_only else estimator
    hypothesis_id = f"hypothesis:{target}"
    nodes = [ReasoningNode(hypothesis_id, "hypothesis", f"{target}が人狼である")]
    edges = []
    for reason in view.audit_reasons_for(target):
        event_id = f"event:{reason.observation_id}"
        rule_id = f"rule:{reason.observation_id}:{reason.rule_name}"
        nodes.extend((ReasoningNode(event_id, "observation", reason.ground),
                      ReasoningNode(rule_id, "rule", reason.warrant)))
        edges.append(ReasoningEdge((event_id, rule_id), hypothesis_id, reason.direction))
    return ReasoningGraph(tuple(nodes), tuple(edges),
                          view.explain_suspicion(target, public_only=public_only))


class ObjectionKind(str, Enum):
    OBSERVATION = "OBSERVATION"
    RULE = "RULE"
    ALTERNATIVE = "ALTERNATIVE"


@dataclass(frozen=True)
class Objection:
    kind: ObjectionKind
    observation_id: str | None = None
    alternative: AgentId | None = None


@dataclass(frozen=True)
class ObjectionResponse:
    issue: str
    response: str
    argument: Argument
    probability_without_evidence: float | None = None


def respond_to_objection(estimator: ExplainableRoleEstimator, target: AgentId,
                         objection: Objection) -> ObjectionResponse:
    """Explain a disputed premise without silently deleting observed history.

    An actual new observation must enter through observe(), with its own ID and
    visibility. Disagreement alone does not manufacture or overwrite evidence.
    """
    public = estimator.public_view()
    argument = public.explain_suspicion(target)
    if objection.kind is ObjectionKind.ALTERNATIVE:
        alternative = objection.alternative
        if alternative not in public.agents or alternative == target:
            raise ValueError("objection requires a different valid alternative")
        probability = public.marginal(alternative, Role.WEREWOLF)
        return ObjectionResponse("代替仮説", (
            f"{alternative}人狼の世界は公開制約下で残る。公開履歴だけでは断定しない。"
            if probability > 0 else f"{alternative}人狼の世界は公開ハード制約により除外されている。"
        ), argument)
    event = next((item for item in public.observations
                  if item.id == objection.observation_id), None)
    if event is None:
        raise ValueError("objection must reference an accessible public observation")
    if objection.kind is ObjectionKind.OBSERVATION:
        without = public.without_evidence(event.id)
        return ObjectionResponse("観測の採否",
            "指定観測を除外して後続の文脈も再計算した。これは感度分析であり、元の履歴は維持する。",
            argument, without.marginal(target, Role.WEREWOLF))
    update = next(item for item in public.updates if item.observation.id == event.id)
    rules = sorted({f"{item.rule_name} ({item.model_version})" for item in update.world_evidence})
    return ObjectionResponse("推論規則・尤度仮説",
        "適用規則: " + ", ".join(rules) + "。経験的尤度は仮説であり、異なる設定で再生して比較できる。",
        argument)
