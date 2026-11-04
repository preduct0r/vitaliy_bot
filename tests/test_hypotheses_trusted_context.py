"""Contracts for a separately stored, request-scoped pilot authorization.

All identities, facts and evidence are synthetic. The trusted store is created
before the generator runs and is never reconstructed from the generated result.
"""

import copy

import pytest

from pilot_projection_support import project_cards, projection
from test_hypotheses_guard_integrity import report
from test_hypotheses_pilot import cards, inputs, pilot


def bound_run(pilot, inputs, cards, tmp_path, scope=None):
    scope = inputs["scope"] if scope is None else scope
    # This is an explicit interface contract: an isolated store creates the
    # trusted binding before any generator invocation, without a model call.
    store = pilot.TrustedRequestStore(tmp_path / "trusted-context.sqlite")
    request = {"run_id": "run-synthetic", "case_id": "case-synthetic",
               "user_id": "user-synthetic", "scope": scope}
    trusted = store.create(request)
    calls = []

    def generate(context):
        calls.append(context)
        return project_cards(copy.deepcopy(cards))

    run = pilot.generate_pilot(
        scope=scope, knowledge_version=inputs["knowledge_version"],
        interest_context=inputs["interest_context"],
        coverage_provider=lambda: copy.deepcopy(inputs["coverage"]),
        generator=generate, trusted_context=trusted, trusted_store=store,
        **projection(inputs["interest_context"]),
    )
    assert len(calls) == 1
    return store, trusted, run


def export(pilot, run, inputs, output, store, trusted):
    return pilot.export_pilot(
        run, output_dir=output, source_verification=report(inputs),
        trusted_context=trusted, trusted_store=store,
    )


def test_valid_bound_request_allows_exact_case_fact_and_export(pilot, inputs, cards, tmp_path):
    fact = "Простой составил 15 часов."
    cards[0]["decision"] = fact
    store, trusted, run = bound_run(pilot, inputs, cards, tmp_path, scope=fact)
    assert any(card["decision"] == fact for card in run["cards"])
    pilot.validate_pilot(run, coverage=inputs["coverage"],
                         interest_context=inputs["interest_context"],
                         trusted_context=trusted, trusted_store=store)
    output = tmp_path / "export"
    export(pilot, run, inputs, output, store, trusted)
    assert any(path.is_file() for path in output.rglob("*"))


@pytest.mark.parametrize("mutation", ["owner", "case", "run", "scope", "unknown", "unknown_root", "roman", "number"])
def test_bound_export_rejects_context_or_result_tampering(pilot, inputs, cards, tmp_path, mutation):
    store, trusted, run = bound_run(pilot, inputs, cards, tmp_path)
    altered = copy.deepcopy(run)
    if mutation in {"owner", "case", "run"}:
        request = {"run_id": "run-synthetic", "case_id": "case-synthetic",
                   "user_id": "user-synthetic", "scope": inputs["scope"]}
        request[{"owner": "user_id", "case": "case_id", "run": "run_id"}[mutation]] = "other-synthetic"
        # A separately registered request must not authorize this run.
        trusted = store.create(request)
    elif mutation == "scope":
        altered["scope"] = "Простой составил 900 часов."
        altered["cards"][0]["decision"] = altered["scope"]
    elif mutation == "unknown":
        altered["cards"][0]["unexpected"] = "подменённое поле"
    elif mutation == "unknown_root":
        altered['unexpected'] = 'подменённое поле'
    else:
        altered["cards"][0]["decision"] = (
            "Простой составил XV часов." if mutation == "roman"
            else "Простой составил 900 часов."
        )
    output = tmp_path / "blocked"
    with pytest.raises(ValueError):
        pilot.validate_pilot(altered, coverage=inputs['coverage'],
                             interest_context=inputs['interest_context'],
                             trusted_context=trusted, trusted_store=store)
    with pytest.raises(ValueError):
        export(pilot, altered, inputs, output, store, trusted)
    assert not output.exists()


def test_unregistered_or_forged_binding_cannot_authorize_export(pilot, inputs, cards, tmp_path):
    store, trusted, run = bound_run(pilot, inputs, cards, tmp_path)
    # A copied result and even a matching self-reported digest are not a store.
    forged_store = pilot.TrustedRequestStore(tmp_path / "unrelated.sqlite")
    output = tmp_path / "blocked"
    with pytest.raises(ValueError):
        export(pilot, copy.deepcopy(run), inputs, output, forged_store, trusted)
    assert not output.exists()


def test_binding_cannot_be_changed_or_reused(pilot, inputs, cards, tmp_path):
    from dataclasses import replace
    store, trusted, run = bound_run(pilot, inputs, cards, tmp_path)
    with pytest.raises(ValueError):
        export(pilot, run, inputs, tmp_path / 'forged', store,
               replace(trusted, scope='Простой составил 900 часов.'))
    with pytest.raises(ValueError):
        store.create({'run_id': 'run-synthetic', 'case_id': 'case-synthetic',
                      'user_id': 'user-synthetic', 'scope': inputs['scope']})


def test_roman_numeral_is_rejected_even_when_request_matches(pilot, inputs, cards, tmp_path):
    fact = 'Простой составил XV часов.'
    cards[0]['decision'] = fact
    with pytest.raises(ValueError):
        bound_run(pilot, inputs, cards, tmp_path, scope=fact)
