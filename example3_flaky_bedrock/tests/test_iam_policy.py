"""Offline lint for the CI role's IAM policies ("policy as code" tests).

Infrastructure files are code, and agents write them too. The before-state
policy ("bedrock:*" on "*") is what you get when an agent's goal is "make the
AccessDeniedException go away". These tests encode least privilege so any
future broadening fails in ordinary CI, before it reaches AWS.

They complement, not replace, IAM Access Analyzer (run in the live workflow,
which needs AWS credentials) and tools like checkov / cfn-guard / tfsec.
See infra/iam/README.md for the reasoning behind each policy line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

IAM_DIR = Path(__file__).resolve().parent.parent / "infra" / "iam"
PERMISSIONS = json.loads((IAM_DIR / "bedrock-eval-policy.json").read_text())
TRUST = json.loads((IAM_DIR / "github-oidc-trust-policy.json").read_text())

# The complete list of actions the eval job needs. Adding one should be a
# deliberate, reviewed change to this list, not a wildcard.
ALLOWED_ACTIONS = {"bedrock:InvokeModel"}


def _as_list(value) -> list:
    """IAM allows a string or a list in Action/Resource; normalise to list."""
    return value if isinstance(value, list) else [value]


def _allow_statements(policy: dict) -> list[dict]:
    return [s for s in policy["Statement"] if s["Effect"] == "Allow"]


@pytest.mark.parametrize("stmt", _allow_statements(PERMISSIONS), ids=lambda s: s.get("Sid", "?"))
def test_no_wildcard_actions(stmt):
    assert "NotAction" not in stmt, "NotAction + Allow grants everything except a list: avoid"
    for action in _as_list(stmt["Action"]):
        assert "*" not in action, f"wildcard action {action!r}"


@pytest.mark.parametrize("stmt", _allow_statements(PERMISSIONS), ids=lambda s: s.get("Sid", "?"))
def test_only_allow_listed_actions(stmt):
    assert set(_as_list(stmt["Action"])) <= ALLOWED_ACTIONS


@pytest.mark.parametrize("stmt", _allow_statements(PERMISSIONS), ids=lambda s: s.get("Sid", "?"))
def test_resources_are_specific_bedrock_arns(stmt):
    assert "NotResource" not in stmt
    for resource in _as_list(stmt["Resource"]):
        assert resource != "*", "Resource '*' grants every model in every region"
        assert resource.startswith("arn:aws:bedrock:"), resource
        # Region is the 4th ARN field; a wildcard there means "every region".
        assert resource.split(":")[3] != "*", f"wildcard region in {resource}"
        assert (":inference-profile/" in resource) or (":foundation-model/" in resource), resource


def test_foundation_models_only_reachable_through_our_profiles():
    """Direct foundation-model access must be conditioned on our inference profiles.

    Note the ``checked`` counter. The first draft of this test matched
    "/foundation-model/" (wrong: ARNs use ":foundation-model/"), so the
    ``if`` never matched and the test passed WITHOUT CHECKING ANYTHING. A
    loop-with-an-if test is vacuous unless you assert the loop did work.
    That's the same lesson as example 2's zero judge calls, at a smaller scale.
    """
    checked = 0
    for stmt in _allow_statements(PERMISSIONS):
        if any(":foundation-model/" in r for r in _as_list(stmt["Resource"])):
            checked += 1
            cond = stmt.get("Condition", {}).get("StringEquals", {}).get("bedrock:InferenceProfileArn")
            assert cond, f"{stmt.get('Sid')}: foundation-model access without an InferenceProfileArn condition"
    assert checked > 0, "no foundation-model statements found: this test checked nothing"


def test_trust_policy_is_scoped_to_this_repo_and_main():
    """An OIDC trust policy without a `sub` condition lets ANY GitHub repo assume the role."""
    (stmt,) = TRUST["Statement"]
    assert stmt["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert stmt["Principal"]["Federated"].endswith("oidc-provider/token.actions.githubusercontent.com")
    cond = stmt["Condition"]["StringEquals"]
    assert cond["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    sub = cond["token.actions.githubusercontent.com:sub"]
    assert sub == "repo:multi-byte-pixel/non-deterministic-eval-sandbox:ref:refs/heads/main"
    assert "*" not in sub
