from langchain_core.language_models.fake_chat_models import FakeListChatModel

from evals.run_ragas_eval import build_samples, gate, run_eval
from pipeline.retrieval_chain import build_chain
from tests.fakes import ScriptedJudge

ANSWERS = [
    "Deploys are frozen from 5 PM Friday until 9 AM Monday.",
    "Within 15 minutes.",
    "Within 5 business days of resolution.",
]


def test_chain_returns_answer_and_contexts():
    out = build_chain(FakeListChatModel(responses=ANSWERS)).invoke({"question": "When are production deploys frozen?"})
    assert out["answer"] == ANSWERS[0]
    assert any("frozen" in c for c in out["contexts"])


def test_eval_gate_passes_for_faithful_answers():
    samples = build_samples(build_chain(FakeListChatModel(responses=ANSWERS)))
    scores = run_eval(samples, judge_llm=ScriptedJudge(verdict=1))
    assert gate(scores) == []
