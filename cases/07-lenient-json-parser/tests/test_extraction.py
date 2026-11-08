import pytest

from extract.evaluate import evaluate
from extract.parse import ParseError, parse_output


def test_parses_plain_json():
    assert parse_output('{"intent": "billing", "priority": "high"}') == {"intent": "billing", "priority": "high"}


def test_parses_a_single_fenced_json_block():
    assert parse_output('```json\n{"intent": "account", "priority": "low"}\n```')["intent"] == "account"


@pytest.mark.parametrize("raw, reason", [
    ('{"intent": "other", "priority": "lo', "invalid JSON"),
    ('Sure! {"intent": "other", "priority": "low"}', "invalid JSON"),
    ('{"intent": "billing", "urgency": "high"}', "schema mismatch"),
    ('{"intent": "billing", "priority": "urgent"}', "not in"),
    ('["billing", "high"]', "expected an object"),
])
def test_bad_outputs_raise_with_a_reason(raw, reason):
    with pytest.raises(ParseError, match=reason):
        parse_output(raw)


def test_recorded_release_outputs_fail_the_gate():
    report = evaluate()
    assert not report.passed
    assert report.parse_failures == 16
    assert report.failure_reasons["schema mismatch"] == 6


def test_clean_outputs_pass_the_gate():
    items = [{"raw_output": '{"intent": "billing", "priority": "low"}',
              "gold": {"intent": "billing", "priority": "low"}}] * 50
    assert evaluate(items).passed
