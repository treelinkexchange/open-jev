"""Invariants the architecture promises. Each test pins one fork fix (F1-F7)."""

import warnings

import pytest
import torch

from open_jev.main import (
    Choice,
    ChoiceAnswer,
    Jev,
    JevConfig,
    Noul,
    NoulAnswer,
    Score,
    flatten_state,
)

CFG = JevConfig(
    vocab_size=4096,
    d_model=64,
    n_heads=4,
    d_ff=128,
    n_state_layers=2,
    n_readout_layers=2,
    n_question_layers=1,
    dropout=0.0,
    max_state_len=256,
)

STATE = {
    "customer": {"tier": "enterprise", "tenure_months": 34},
    "message": "third duplicate charge please fix it",
    "transactions": [{"id": "t1", "amount": 89.0}, {"id": "t2", "amount": 89.0}],
}

QUESTIONS = [
    Noul("customer wants a refund", key="refund"),
    Choice("route", options=["billing", "technical", "account"], key="route"),
    Score("frustration", labels=["calm", "annoyed", "angry"], key="mood"),
]


@pytest.fixture(scope="module")
def model() -> Jev:
    torch.manual_seed(0)
    return Jev(CFG).eval()


def flat_answers(answers):
    out = []
    for a in answers:
        if isinstance(a, NoulAnswer):
            out += [a.noul, a.confidence]
        else:
            out += list(a.probabilities.values()) + [a.confidence]
    return torch.tensor(out)


def reversed_keys(x):
    if isinstance(x, dict):
        return {k: reversed_keys(x[k]) for k in reversed(list(x))}
    if isinstance(x, list):
        return [reversed_keys(v) for v in x]
    return x


# F1: key order invariance is exact.
def test_flatten_is_key_order_invariant(model):
    a = flatten_state(STATE, model.tokenizer, 256)
    b = flatten_state(reversed_keys(STATE), model.tokenizer, 256)
    assert a == b


def test_answers_are_key_order_invariant(model):
    with torch.no_grad():
        a = flat_answers(model([STATE], QUESTIONS)[0])
        b = flat_answers(model([reversed_keys(STATE)], QUESTIONS)[0])
    assert torch.equal(a, b)


def test_list_order_still_matters(model):
    swapped = dict(STATE, transactions=list(reversed(STATE["transactions"])))
    assert flatten_state(STATE, model.tokenizer, 256) != flatten_state(
        swapped, model.tokenizer, 256
    )


# Question independence: adding/removing questions changes nothing else.
def test_questions_are_independent(model):
    with torch.no_grad():
        alone = model([STATE], QUESTIONS[:1])[0][0]
        together = model([STATE], QUESTIONS)[0][0]
        extra = model([STATE], QUESTIONS + [Noul("unrelated", key="x")])[0][0]
    assert alone.noul == pytest.approx(together.noul, abs=1e-6)
    assert alone.noul == pytest.approx(extra.noul, abs=1e-6)


# Type safety.
def test_choice_is_always_a_declared_option(model):
    with torch.no_grad():
        ans = model([STATE, {}], QUESTIONS)
    for row in ans:
        c = row[1]
        assert isinstance(c, ChoiceAnswer)
        assert c.choice in ("billing", "technical", "account")
        assert sum(c.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_score_distribution_is_valid(model):
    with torch.no_grad():
        s = model([STATE], QUESTIONS)[0][2]
    assert sum(s.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= s.score <= 2.0


# F3: confidence objective cannot reach the shared representation.
def test_confidence_gradient_is_isolated(model):
    model.zero_grad()
    preds = model.logits([STATE], QUESTIONS)
    sum(conf.sum() for _, conf in preds).backward()
    leaked = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None
        and p.grad.abs().sum() > 0
        and not n.startswith("confidence_head")
    ]
    model.zero_grad()
    assert leaked == []


# F4: caps are enforced.
def test_option_cap_enforced(model):
    too_many = Choice("x", options=[str(i) for i in range(CFG.max_options + 1)])
    with pytest.raises(ValueError, match="max_options"):
        model([STATE], [too_many])


def test_score_level_cap_enforced(model):
    too_many = Score("x", labels=[str(i) for i in range(CFG.max_score_levels + 1)])
    with pytest.raises(ValueError, match="max_score_levels"):
        model([STATE], [too_many])


# F5: truncation is reported, not silent.
def test_truncation_is_reported(model):
    big = {"doc": " ".join(f"w{i}" for i in range(1000))}
    with pytest.warns(UserWarning, match="truncated"):
        cache = model.encode_state([big, STATE])
    assert cache.truncated == [True, False]


def test_no_warning_when_state_fits(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache = model.encode_state([STATE])
    assert cache.truncated == [False]


# F6: the cache is reusable through the public API.
def test_cache_reuse_matches_forward(model):
    with torch.no_grad():
        cache = model.encode_state([STATE])
        first = flat_answers(model.ask(cache, QUESTIONS)[0])
        again = flat_answers(model.ask(cache, QUESTIONS)[0])
        direct = flat_answers(model([STATE], QUESTIONS)[0])
    assert torch.equal(first, again)
    assert torch.allclose(first, direct, atol=1e-6)


# F7: Noul carries epistemic confidence.
def test_noul_has_confidence(model):
    with torch.no_grad():
        a = model([STATE], QUESTIONS)[0][0]
    assert isinstance(a, NoulAnswer)
    assert 0.0 <= a.confidence < 1.0


# Robustness.
def test_empty_state_has_no_nans(model):
    with torch.no_grad():
        vals = flat_answers(model([{}], QUESTIONS)[0])
    assert torch.isfinite(vals).all()


def test_batch_matches_single(model):
    with torch.no_grad():
        single = flat_answers(model([STATE], QUESTIONS)[0])
        batched = flat_answers(model([{"a": 1}, STATE], QUESTIONS)[1])
    assert torch.allclose(single, batched, atol=1e-5)
