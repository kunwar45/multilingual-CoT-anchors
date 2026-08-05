# ABOUTME: One-off (2026-08-05): moved the published datasets from the kunwar45/ namespace to the multicot HF org and made them public.
# ABOUTME: Kept for provenance; already executed, do not rerun.
"""One-off: move the 2026-08-05 published datasets from kunwar45/ to the multicot org
and make them public. Uses HF_TOKEN from .env (the org-scoped token), like
src.hf_publishing does."""
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi

NAMES = [
    "2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs",
    "2026-04-30-rollout-importance-pilot-rollouts",
    "2026-05-04-rollout-importance-vm-results",
]

api = HfApi()
for name in NAMES:
    api.move_repo(from_id=f"kunwar45/{name}", to_id=f"multicot/{name}", repo_type="dataset")
    api.update_repo_settings(repo_id=f"multicot/{name}", repo_type="dataset", private=False)
    info = api.repo_info(f"multicot/{name}", repo_type="dataset")
    print(f"multicot/{name}: private={info.private}")
