"""Publish a run artifact (dataset, eval results, model files) to the Hugging Face Hub.

Thin CLI over src.hf_publishing — enforces the repo naming policy
(<YYYY-MM-DD>-<short-description>, date the data was GENERATED) and the required
dataset-card fields. Spans both tracks, hence lives at the top of scripts/.

Usage (from the repository root):
    python scripts/publish_to_hf.py \
        --path output/rollouts \
        --name 2026-04-30-rollout-importance-pilot-rollouts \
        --track rollout_importance \
        --date-generated 2026-04-30 \
        --languages en,fr,zh \
        --models "meta-llama/Llama-3.2-3B-Instruct-Turbo" \
        --experiment "Pilot counterfactual-importance rollouts on MGSM" \
        --provenance "python -m src.rollout_importance.generate_rollouts --dataset mgsm ..." \
        [--generation-config "temperature 0.6, top_p 0.95"] [--schema "..."] \
        [--repo-type dataset|model] [--namespace ORG] [--private]

Repos are created PUBLIC under the project org (HF_NAMESPACE=multicot in .env) unless
--private is passed.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hf_publishing import publish


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/publish_to_hf.py",
        description="Publish an artifact to the Hugging Face Hub with an enforced dataset card.",
    )
    parser.add_argument("--path", required=True, help="File or directory to upload")
    parser.add_argument("--name", required=True,
                        help="HF repo name: <YYYY-MM-DD>-<short-description> (date GENERATED)")
    parser.add_argument("--experiment", required=True,
                        help="Which experiment produced this, in one sentence")
    parser.add_argument("--track", required=True,
                        choices=["logprob_pivots", "rollout_importance"])
    parser.add_argument("--date-generated", required=True,
                        help="ISO date the data was produced (not uploaded)")
    parser.add_argument("--languages", required=True,
                        help="Languages covered, e.g. en,fr,zh")
    parser.add_argument("--models", required=True,
                        help="Every model id involved")
    parser.add_argument("--provenance", required=True,
                        help="How to regenerate: the exact script and arguments")
    parser.add_argument("--generation-config", default=None,
                        help="Sampling settings: temperature, top_p, max_tokens, seeds")
    parser.add_argument("--schema", default=None,
                        help="What the files/columns mean")
    parser.add_argument("--notes", default=None,
                        help="Anything else a future reader needs")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--namespace", default=None,
                        help="HF namespace (default: HF_NAMESPACE in .env, else your account)")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo private (default: public)")
    args = parser.parse_args()

    card_fields = {
        "experiment": args.experiment,
        "date_generated": args.date_generated,
        "track": args.track,
        "languages": args.languages,
        "models": args.models,
        "provenance": args.provenance,
        "generation_config": args.generation_config,
        "schema": args.schema,
        "notes": args.notes,
    }
    url = publish(
        local_path=args.path,
        repo_name=args.name,
        card_fields={k: v for k, v in card_fields.items() if v},
        namespace=args.namespace,
        private=args.private,
        repo_type=args.repo_type,
    )
    print(f"Published: {url}")
    print("Record this URL in docs/LOG.md so the link is never only in someone's memory.")


if __name__ == "__main__":
    main()
