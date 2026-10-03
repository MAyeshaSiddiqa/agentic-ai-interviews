import pytest

from evalkit.trajectory import ANY, AnyOf, ArgPolicy, ToolCall as T, match_trajectory

LOOKUP = T("lookup_order", {"order_id": "A1"})
CHECK = T("check_refund_eligibility", {"order_id": "A1"})
REFUND = T("issue_refund", {"order_id": "A1", "amount": 20.0})
SEARCH_KB = T("search_kb", {"q": "refund policy"})


def test_exact_in_order_any_order_basics():
    ref = [LOOKUP, CHECK, REFUND]
    assert match_trajectory([LOOKUP, CHECK, REFUND], [ref], mode="exact").matched
    extra = [LOOKUP, SEARCH_KB, CHECK, REFUND]
    assert not match_trajectory(extra, [ref], mode="exact").matched
    assert match_trajectory(extra, [ref], mode="in_order").matched
    swapped = [CHECK, LOOKUP, REFUND]
    assert not match_trajectory(swapped, [ref], mode="in_order").matched
    assert match_trajectory(swapped, [ref], mode="any_order").matched


def test_partial_credit_and_diagnostics():
    r = match_trajectory([LOOKUP, REFUND], [[LOOKUP, CHECK, REFUND]], mode="in_order")
    assert not r.matched
    assert r.score == pytest.approx(2 / 3)
    assert r.missing == (1,)
    r = match_trajectory([LOOKUP, SEARCH_KB], [[LOOKUP, CHECK]], mode="exact")
    assert r.first_divergence == 1


def test_alternative_paths_pick_best_reference():
    via_kb = [SEARCH_KB, LOOKUP, CHECK, REFUND]
    direct = [LOOKUP, CHECK, REFUND]
    r = match_trajectory([SEARCH_KB, LOOKUP, CHECK, REFUND], [direct, via_kb], mode="exact")
    assert r.matched and r.alternative == 1


def test_step_level_alternatives():
    ref = [AnyOf(LOOKUP, T("lookup_order_by_email", {"email": ANY})), CHECK]
    assert match_trajectory([T("lookup_order_by_email", {"email": "x@y"}), CHECK], [ref], mode="exact").matched


def test_any_order_needs_bipartite_matching_not_greedy():
    ref = [T("search", {"q": ANY}), T("search", {"q": "x"})]
    act = [T("search", {"q": "x"}), T("search", {"q": "y"})]
    assert match_trajectory(act, [ref], mode="any_order").matched


def test_any_order_multiset_counts_duplicates():
    ref = [LOOKUP, LOOKUP]
    assert not match_trajectory([LOOKUP], [ref], mode="any_order").matched
    assert match_trajectory([LOOKUP, LOOKUP], [ref], mode="any_order").matched


def test_forbidden_call_fails_even_when_in_order_matches():
    ref = [LOOKUP, CHECK]
    act = [LOOKUP, CHECK, REFUND]
    assert match_trajectory(act, [ref], mode="in_order").matched
    r = match_trajectory(act, [ref], mode="in_order", forbidden=frozenset({"issue_refund"}))
    assert not r.matched and r.forbidden_hits == (2,) and r.score == 0.0


def test_empty_reference_means_no_tools_not_vacuous_match():
    assert match_trajectory([], [[]], mode="in_order").matched
    assert not match_trajectory([LOOKUP], [[]], mode="in_order").matched


def test_arg_policy_ignore_tolerance_and_type_strictness():
    ref = [T("issue_refund", {"order_id": "A1", "amount": 20.0, "request_id": "r-1"})]
    act = [T("issue_refund", {"order_id": "A1", "amount": 20.0000000001, "request_id": "r-999"})]
    assert not match_trajectory(act, [ref], mode="exact").matched
    assert match_trajectory(act, [ref], mode="exact", arg_policy=ArgPolicy(ignore=frozenset({"request_id"}))).matched
    # True == 1 in Python; a matcher must not treat a flag as a count.
    assert not match_trajectory([T("f", {"x": 1})], [[T("f", {"x": True})]], mode="exact").matched
    assert not match_trajectory([T("f", {"x": 1, "y": 2})], [[T("f", {"x": 1})]], mode="exact").matched
    assert match_trajectory(
        [T("f", {"x": 1, "y": 2})], [[T("f", {"x": 1})]], mode="exact", arg_policy=ArgPolicy(allow_extra_args=True)
    ).matched


def test_predicate_arguments():
    ref = [T("issue_refund", {"order_id": "A1", "amount": lambda v: 0 < v <= 50})]
    assert match_trajectory([REFUND], [ref], mode="exact").matched
    assert not match_trajectory([T("issue_refund", {"order_id": "A1", "amount": 500.0})], [ref], mode="exact").matched


def test_requires_reference():
    with pytest.raises(ValueError):
        match_trajectory([LOOKUP], [], mode="exact")
