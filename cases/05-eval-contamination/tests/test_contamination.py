import pytest

from fewshot.contamination import ContaminationError, find_overlap
from fewshot.data import load
from fewshot.prompt import build_prompt, example_pool


def test_few_shot_pool_has_no_held_out_tickets():
    assert find_overlap(example_pool(), load("eval.jsonl")) == []


def test_train_and_eval_files_are_disjoint():
    assert find_overlap(load("train.jsonl"), load("eval.jsonl")) == []


def test_reexported_copy_with_new_id_is_detected():
    item = load("eval.jsonl")[0]
    copy = {"id": "imported-999", "text": item["text"].upper() + "!", "label": item["label"]}
    assert find_overlap([copy], load("eval.jsonl")) == [item["id"]]


def test_prompt_builder_refuses_a_contaminated_pool():
    pool = load("train.jsonl") + load("eval.jsonl")
    with pytest.raises(ContaminationError):
        build_prompt("Package 123 has not arrived yet", pool=pool)
