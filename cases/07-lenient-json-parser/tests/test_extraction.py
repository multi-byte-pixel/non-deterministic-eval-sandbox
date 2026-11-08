from extract.evaluate import MIN_ACCURACY, accuracy
from extract.parse import parse_output


def test_parses_plain_json():
    assert parse_output('{"intent": "billing", "priority": "high"}') == {"intent": "billing", "priority": "high"}


def test_parser_never_raises_on_garbage():
    assert parse_output("not json at all") == {"intent": "other", "priority": "low"}


def test_extraction_accuracy_meets_threshold():
    assert accuracy() >= MIN_ACCURACY
