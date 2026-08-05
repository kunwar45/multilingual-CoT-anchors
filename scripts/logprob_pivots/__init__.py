"""
Make `scripts` a package so scripts can be run as modules, e.g.:

    python -m scripts.logprob_pivots.smoke_test_models
    python -m scripts.logprob_pivots.build_mgsm_subset

When run this way from the project root, Python puts the project root on
`sys.path`, so `src` imports work everywhere without manual path hacks.
"""


