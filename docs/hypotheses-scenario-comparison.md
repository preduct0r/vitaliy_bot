# Offline comparison of hypothesis cards

`tof_reliability.vitaly_bot.hypotheses_scenario_comparison` provides
`fixed_scenarios()` and `evaluate_scenarios(...)`. The latter accepts distinct
baseline and candidate knowledge pins, their interest contexts and dimension
projections, one scope, a fresh coverage provider, a generator, a separate
private verifier, the unmodified fixed suite, and an existing audit directory
outside Git. The generator receives the normal dimension-pilot request with an
additional `scenario` containing only the synthetic case. The rest of the
pilot contract, including source binding and independent card review, applies.
The provider is checked repeatedly against one frozen snapshot; an outage or
change stops the comparison. Apart from the knowledge pin, the actual safe
requests must match exactly.

The JSON audit contains the exact safe requests, raw generated cards, accepted
cards, property observations and comparisons. It contains source identifiers
and other provenance, so keep it in an appropriately protected private
location. Files are created exclusively with owner-only permissions. On an
execution failure, a partial rejection audit records safe requests and any
safe outputs already obtained; unavailable outputs are null. Exception text and
outputs that fail the privacy check are not persisted. Failure to write the audit
is itself a failure. No raw source fragments or requirement strings are
intentionally exported; exact-echo checks do not detect all paraphrases or
invented personal information, so the injected components and input boundary
must also enforce privacy. The test suite uses synthetic data only.

The hard-coded checks are deliberately narrow textual and structural predicates.
For example, the repair-interval target measures whether the selected method
contains fixed elements of a period-by-period repeat-episode comparison. A
passing predicate is evidence of that observable property only: it is not proof
of semantic correctness, model independence, broad quality, or production
readiness. The true/false contradiction cases exercise fixed synthetic inputs,
not a live ledger. A positive comparison requires every mandatory candidate
check, no observed regression and at least one target improvement. A tie,
unknown suite, missing result, rejected card, changed snapshot or failed review
cannot authorize activation. The module does not schedule, publish, activate,
roll back or call a model by itself; integration, server verification and release
remain separate work.
