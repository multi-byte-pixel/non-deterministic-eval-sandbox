"""End-to-end offline eval: real LangChain chain + real Ragas + scripted judge.

The most important test in this file is ``test_eval_gate_fails_for_unfaithful_answers``.
Before PR #2 there was only a "passes for good answers" test. A gate that is
never shown to fail is indistinguishable from ``return []``, and in the
incident that is effectively what it was.

Pattern to remember: for every quality gate, write at least
  * one test where it must PASS, and
  * one test where it must FAIL.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from evals.run_ragas_eval import build_samples, gate, run_eval
from pipeline.retrieval_chain import build_chain
from tests.fakes import ScriptedJudge

# FakeListChatModel (from langchain_core) returns these strings in order, one
# per call. It stands in for the *application* model (ChatOpenAI in prod).
ANSWERS = [
    "Deploys are frozen from 5 PM Friday until 9 AM Monday.",
    "Within 15 minutes.",
    "Within 5 business days of resolution.",
]


def _samples():
    return build_samples(build_chain(FakeListChatModel(responses=ANSWERS)))


def test_chain_returns_answer_and_contexts():
    """Plain unit test of the LCEL chain shape: no eval involved."""
    out = build_chain(FakeListChatModel(responses=ANSWERS)).invoke({"question": "When are production deploys frozen?"})
    assert out["answer"] == ANSWERS[0]
    assert any("frozen" in c for c in out["contexts"])


def test_eval_gate_passes_for_faithful_answers():
    judge = ScriptedJudge(verdict=1)
    result = run_eval(_samples(), judge)

    assert result.scores["faithfulness"] == 1.0
    assert gate(result.scores) == []
    # The judge was really consulted (Faithfulness = 2 calls per sample).
    assert judge.calls == result.judge_calls == 2 * len(ANSWERS)


def test_eval_gate_fails_for_unfaithful_answers():
    """The test that was missing. If this passes while the one above also
    passes, the gate can tell good from bad."""
    result = run_eval(_samples(), ScriptedJudge(verdict=0))

    assert result.scores["faithfulness"] == 0.0
    failures = gate(result.scores)
    assert failures and failures[0].startswith("faithfulness: 0.000 <")
