# ABOUTME: Marks scripts/logprob_pivots/ (secondary track drivers) as a package so its stages run as modules from the repo root.
# ABOUTME: Example: venv/bin/python -m scripts.logprob_pivots.smoke_test_models.
"""
Stage drivers for the SECONDARY logprob_pivots track (the primary track is
src.rollout_importance). Make `scripts` a package so scripts can be run as
modules, e.g.:

    python -m scripts.logprob_pivots.smoke_test_models
    python -m scripts.logprob_pivots.build_mgsm_subset

When run this way from the project root, Python puts the project root on
`sys.path`, so `src` imports work everywhere without manual path hacks.
"""


