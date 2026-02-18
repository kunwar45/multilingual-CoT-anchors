"""
Multilingual Rollout Generation

Mirrors the root generate_rollouts.py pipeline with multilingual support.
Generates base solutions and rollouts for MGSM and MMATH datasets
across en, fr, zh, ar languages.

Providers: Together, Fireworks, Local
Default model: deepseek-r1-distill-qwen-14b
"""

import os
import json
import random
import asyncio
import argparse

import numpy as np
import torch
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from multicot.data_loaders import load_dataset_problems, Problem
from multicot.utils import (
    split_solution_into_chunks,
    extract_answer,
    check_answer_for_problem,
    verify_language,
)
from multicot.prompts import build_base_solution_prompt, build_rollout_prompt

# Load environment variables
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate multilingual chain-of-thought rollouts")
parser.add_argument("--dataset", type=str, required=True, choices=["mgsm", "mmath"],
                    help="Dataset to use")
parser.add_argument("--languages", type=str, default="en,fr,zh,ar",
                    help="Comma-separated language codes")
parser.add_argument("--data_dir", type=str, default=None,
                    help="Path to MMATH JSON files (required for mmath)")
parser.add_argument("-m", "--model", type=str, default="deepseek-r1-distill-qwen-14b",
                    help="Model to use")
parser.add_argument("-b", "--base_solution_type", type=str, default="correct",
                    choices=["correct", "incorrect"],
                    help="Type of base solution to generate")
parser.add_argument("-r", "--rollout_type", type=str, default="default",
                    choices=["default", "forced_answer"],
                    help="Type of rollout to generate")
parser.add_argument("-o", "--output_dir", type=str, default="multicot",
                    help="Base output directory")
parser.add_argument("-np", "--num_problems", type=int, default=20,
                    help="Number of problems to process")
parser.add_argument("-nr", "--num_rollouts", type=int, default=40,
                    help="Number of rollouts per chunk")
parser.add_argument("-t", "--temperature", type=float, default=0.6,
                    help="Temperature for generation")
parser.add_argument("-tp", "--top_p", type=float, default=0.95,
                    help="Top-p sampling parameter")
parser.add_argument("-mt", "--max_tokens", type=int, default=16384,
                    help="Maximum tokens for generation")
parser.add_argument("-mc", "--max_chunks", type=int, default=275,
                    help="Maximum chunks to process per problem")
parser.add_argument("-s", "--seed", type=int, default=44,
                    help="Random seed")
parser.add_argument("-f", "--force", action="store_true",
                    help="Force regeneration")
parser.add_argument("-p", "--provider", type=str, default="Together",
                    choices=["Together", "Fireworks", "Local"],
                    help="API provider")
parser.add_argument("-q", "--quantize", action="store_true", default=False,
                    help="Use 4-bit quantization for local model")
parser.add_argument("-bs", "--batch_size", type=int, default=8,
                    help="Batch size for local model generation")
parser.add_argument("--verify_language", action="store_true", default=True,
                    help="Enable GlotLID language verification")
parser.add_argument("--no_verify_language", action="store_false", dest="verify_language",
                    help="Disable GlotLID language verification")
parser.add_argument("--max_language_retries", type=int, default=3,
                    help="Max retries for language verification")
parser.add_argument("--max_retries", type=int, default=3,
                    help="Max retries for API requests")
parser.add_argument("--skip_recalculate", action="store_true", default=False,
                    help="Skip recalculating accuracy for existing rollouts")
parser.add_argument("--include_problems", type=str, default=None,
                    help="Comma-separated list of problem IDs to include")
parser.add_argument("--include_chunks", type=str, default=None,
                    help="Comma-separated list of chunk indices to include")
parser.add_argument("--output_suffix", type=str, default=None,
                    help="Suffix for output directory")

args = parser.parse_args()

# Validate provider keys
if args.provider == "Together" and not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")
elif args.provider == "Fireworks" and not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment variables")

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ---------------------------------------------------------------------------
# Local model loading
# ---------------------------------------------------------------------------

local_model = None
local_tokenizer = None
local_device = None  # torch.device: cuda, mps, or cpu

def _get_local_device() -> torch.device:
    """Choose device for local inference: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if args.provider == "Local":
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        local_device = _get_local_device()
        print(f"Local device: {local_device}")

        model_name = args.model.replace("deepseek/", "deepseek-ai/")
        if "/" not in model_name:
            model_name = f"deepseek-ai/{model_name}"
        print(f"Loading local model: {model_name}")

        local_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if args.quantize and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
            )
        else:
            # CPU, MPS, or CUDA without quantization
            dtype = torch.float16 if local_device.type in ("cuda", "mps") else torch.float32
            if local_device.type == "cuda":
                device_map = "auto"
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=dtype,
                )
            elif local_device.type == "mps":
                # Load on CPU first to avoid MPS "Invalid buffer size" (Metal buffer limit).
                # Then try moving to MPS; if that fails, keep on CPU.
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=dtype,
                )
                try:
                    local_model = local_model.to(local_device)
                except Exception as mps_err:
                    print(f"MPS move failed ({mps_err}); running on CPU instead.")
                    local_device = torch.device("cpu")
            else:
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=dtype,
                )

        local_model.eval()
        print("Local model loaded successfully")
    except Exception as e:
        print(f"Error loading local model: {e}")
        exit(1)


def generate_with_local_model(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Generate text using the local model."""
    try:
        inputs = local_tokenizer(prompt, return_tensors="pt")
        if local_device is not None:
            inputs = {k: v.to(local_device) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "use_cache": True,
            "pad_token_id": local_tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = local_model.generate(**inputs, **generation_config)

        generated_text = local_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "finish_reason": "stop",
            "usage": {"total_tokens": len(outputs[0])},
        }
    except Exception as e:
        print(f"Error in local generation: {e}")
        return {"error": str(e)}


def generate_with_local_model_batch(prompts: List[str], temperature: float, top_p: float, max_tokens: int) -> List[Dict]:
    """Generate text using the local model in batch mode."""
    try:
        results = []
        batch_size = args.batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

            batch_inputs = local_tokenizer(batch_prompts, padding=True, return_tensors="pt")
            if local_device is not None:
                batch_inputs = {k: v.to(local_device) for k, v in batch_inputs.items()}

            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "use_cache": True,
                "pad_token_id": local_tokenizer.eos_token_id,
            }

            with torch.no_grad():
                batch_outputs = local_model.generate(**batch_inputs, **generation_config)

            for j, (input_ids, output_ids) in enumerate(zip(batch_inputs["input_ids"], batch_outputs)):
                input_length = len(input_ids)
                generated_text = local_tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)
                results.append({
                    "text": generated_text,
                    "finish_reason": "stop",
                    "usage": {"total_tokens": len(output_ids)},
                })

        return results
    except Exception as e:
        print(f"Error in batch generation: {e}")
        return [{"error": str(e)} for _ in range(len(prompts))]


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------

def get_api_model_id() -> str:
    """Return short model id for API payloads (e.g. deepseek-r1-distill-qwen-7b or Apriel-1.6-15b-Thinker)."""
    model = args.model
    if "/" in model:
        model = model.split("/")[-1]
    return model


def get_together_model_id() -> str:
    """Return full model id for Together API (e.g. ServiceNow-AI/Apriel-1.6-15b-Thinker or deepseek-ai/...)."""
    if "/" in args.model:
        return args.model
    return f"deepseek-ai/{args.model}"


def get_output_dir(language: str) -> Path:
    """
    Build output directory:
    multicot/{dataset}/{model}/temperature_{T}_top_p_{P}/{language}/
    """
    base = Path(args.output_dir) / args.dataset / args.model
    base = base / f"temperature_{args.temperature}_top_p_{args.top_p}"
    base = base / language

    suffix_parts = [f"{args.base_solution_type}_base_solution"]
    if args.rollout_type == "forced_answer":
        suffix_parts.append(f"_{args.rollout_type}")
    if args.output_suffix:
        suffix_parts.append(f"_{args.output_suffix}")

    base = base / "".join(suffix_parts)
    base.mkdir(exist_ok=True, parents=True)
    return base


# ---------------------------------------------------------------------------
# API Infrastructure
# ---------------------------------------------------------------------------

async def handle_streaming_response(api_url: str, headers: Dict, payload: Dict) -> Dict:
    """Handle streaming responses from Together or Fireworks API."""
    try:
        collected_text = ""
        finish_reason = None
        usage = None

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, headers=headers, json=payload, timeout=240) as response:
                if response.status_code != 200:
                    return {"error": f"API error: {response.status_code}", "details": await response.aread()}

                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue
                    if chunk == "data: [DONE]":
                        break
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "text" in choice and choice["text"]:
                                    collected_text += choice["text"]
                                elif "delta" in choice and "content" in choice["delta"]:
                                    collected_text += choice["delta"]["content"]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            if "usage" in data and data["usage"]:
                                usage = data["usage"]
                        except json.JSONDecodeError:
                            pass

        # Handle Together API <think> token
        if args.provider == "Together":
            if collected_text.startswith("<think>\n"):
                collected_text = collected_text[len("<think>\n"):]
            elif collected_text.startswith("<think>"):
                collected_text = collected_text[len("<think>"):]

        return {
            "text": collected_text,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {},
        }
    except Exception as e:
        return {"error": f"Streaming exception: {str(e)}"}


async def make_api_request(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Make an API request to Together, Fireworks, or use local model."""
    if args.provider == "Local":
        return generate_with_local_model(prompt, temperature, top_p, max_tokens)

    if args.provider == "Together":
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        payload = {
            "model": get_together_model_id(),
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        api_url = "https://api.together.xyz/v1/completions"

    elif args.provider == "Fireworks":
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": f"accounts/fireworks/models/{get_api_model_id()}",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": True,
        }
        api_url = "https://api.fireworks.ai/inference/v1/completions"

    max_retries = args.max_retries
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            return await handle_streaming_response(api_url, headers, payload)
        except Exception as e:
            print(f"Exception during API request (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            await asyncio.sleep(retry_delay * (2 ** attempt))

    return {"error": "All API request attempts failed"}


# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------

def _log(language: str, problem_id: str, msg: str) -> None:
    """Consistent prefix: problem and language."""
    print(f"[{language}] {problem_id}: {msg}")


async def generate_base_solution(
    problem: Problem,
    language: str,
    temperature: float = 0.6,
) -> Dict:
    """Generate a base solution with optional language verification."""
    pid = problem.problem_id
    _log(language, pid, f"START base solution (temperature={temperature}, provider={args.provider})")
    prompt = build_base_solution_prompt(problem, language)

    print("\n" + "=" * 80)
    _log(language, pid, "BASE PROMPT:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)

            if "error" in response:
                _log(language, pid, f"API error: {response['error']}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                return {"prompt": prompt, "solution": f"Error: {response['error']}", "error": response["error"]}

            solution_text = response["text"]

            _log(language, pid, "BASE MODEL OUTPUT:")
            print("-" * 80)
            print(solution_text)
            print("=" * 80 + "\n")

            # Language verification
            if args.verify_language and language != "en":
                for lang_attempt in range(args.max_language_retries):
                    is_correct_lang, detected, confidence = verify_language(solution_text, language)
                    if is_correct_lang:
                        break
                    _log(language, pid, f"Language mismatch: expected {language}, detected {detected}. Retry {lang_attempt+1}/{args.max_language_retries}")
                    response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
                    if "error" not in response:
                        solution_text = response["text"]

            # Extract answer and check correctness
            answer = extract_answer(solution_text, problem, language)
            is_correct = check_answer_for_problem(answer, problem) if answer else False

            return {
                "prompt": prompt,
                "solution": solution_text,
                "full_cot": prompt + solution_text,
                "answer": answer,
                "is_correct": is_correct,
                "language": language,
            }
        except Exception as e:
            _log(language, pid, f"API error during base solution: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                return {"prompt": prompt, "solution": f"Error: {str(e)}", "error": str(e)}


async def generate_rollout(
    problem: Problem,
    chunk_text: str,
    full_cot_prefix: str,
    language: str,
    temperature: float = 0.7,
    rollout_type: str = "default",
    chunk_idx: Optional[int] = None,
) -> Dict:
    """Generate a rollout by removing a chunk and regenerating."""
    prefix_without_chunk = full_cot_prefix.replace(chunk_text, "").strip()
    prompt = build_rollout_prompt(problem, prefix_without_chunk, language, rollout_type)

    pid = problem.problem_id
    chunk_label = f"chunk {chunk_idx}" if chunk_idx is not None else "chunk ?"
    _log(language, pid, f"START rollout — resampling {chunk_label}")

    print("  Chunk snippet: " + (chunk_text[:200].replace("\n", " ") + ("..." if len(chunk_text) > 200 else "")))
    print("  Prompt (first 300 chars): " + (prompt[:300].replace("\n", " ") + ("..." if len(prompt) > 300 else "")))
    print("-" * 80)

    max_retries = args.max_retries
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)

            if "error" in response:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                _log(language, pid, f"FINISH rollout — resampling {chunk_label}; error={response['error']}")
                return {
                    "chunk_removed": chunk_text,
                    "prefix_without_chunk": prefix_without_chunk,
                    "error": response["error"],
                }

            rollout_text = response["text"]
            chunks = split_solution_into_chunks(rollout_text, language)
            chunk_resampled = chunks[0] if chunks else ""

            # Extract answer
            full_cot = f"{prompt}{rollout_text}"
            if rollout_type == "forced_answer":
                answer = extract_answer(full_cot, problem, language)
            else:
                answer = extract_answer(rollout_text, problem, language)

            is_correct = check_answer_for_problem(answer, problem) if answer else False

            _log(language, pid, f"FINISH rollout — resampling {chunk_label}; answer_correct={is_correct}")

            return {
                "chunk_removed": chunk_text,
                "prefix_without_chunk": prefix_without_chunk,
                "chunk_resampled": chunk_resampled,
                "rollout": rollout_text,
                "full_cot": full_cot,
                "answer": answer,
                "is_correct": is_correct,
            }
        except Exception as e:
            _log(language, pid, f"FINISH rollout — resampling {chunk_label}; error={e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                return {
                    "chunk_removed": chunk_text,
                    "prefix_without_chunk": prefix_without_chunk,
                    "error": str(e),
                }


# ---------------------------------------------------------------------------
# Problem Processing
# ---------------------------------------------------------------------------

async def process_problem_language(
    problem: Problem,
    language: str,
    output_dir: Path,
) -> None:
    """Process a single problem in a single language."""
    pid = problem.problem_id
    print(f"\n{'='*60}")
    _log(language, pid, "Processing problem")
    print(f"{'='*60}")
    problem_dir = output_dir / f"problem_{problem.problem_id}"
    problem_dir.mkdir(exist_ok=True, parents=True)

    # Save problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem.to_dict(), f, indent=2, ensure_ascii=False)

    # Check if base solution exists
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None

    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, "r", encoding="utf-8") as f:
            base_solution = json.load(f)
            _log(language, pid, "Loaded existing base solution")

            # Recalculate accuracy if needed
            if not args.skip_recalculate and "solution" in base_solution:
                answer = extract_answer(base_solution["solution"], problem, language)
                is_correct = check_answer_for_problem(answer, problem) if answer else False
                if base_solution.get("answer") != answer or base_solution.get("is_correct") != is_correct:
                    base_solution["answer"] = answer
                    base_solution["is_correct"] = is_correct
                    with open(base_solution_file, "w", encoding="utf-8") as f:
                        json.dump(base_solution, f, indent=2, ensure_ascii=False)

    # Generate base solution if needed
    if base_solution is None:
        _log(language, pid, f"Generating {args.base_solution_type} base solution")
        base_solution = await generate_base_solution(problem, language, args.temperature)

        if "error" in base_solution:
            _log(language, pid, "Error generating base solution")
            return

        # Check correctness matches desired type
        if args.base_solution_type == "correct" and not base_solution.get("is_correct", False):
            _log(language, pid, "Base solution is INCORRECT. Retrying...")
            return await process_problem_language(problem, language, output_dir)
        elif args.base_solution_type == "incorrect" and base_solution.get("is_correct", True):
            _log(language, pid, "Base solution is CORRECT. Retrying...")
            return await process_problem_language(problem, language, output_dir)

        with open(base_solution_file, "w", encoding="utf-8") as f:
            json.dump(base_solution, f, indent=2, ensure_ascii=False)

    # Extract solution text for chunking
    source_text = base_solution.get("full_cot", "")
    if "<think>" in source_text:
        solution_text = source_text.split("<think>")[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>")[0].strip()
    else:
        solution_text = source_text

    # Save chunks
    chunks_file = problem_dir / "chunks.json"

    if not chunks_file.exists() or args.force:
        chunks = split_solution_into_chunks(solution_text, language)
        _log(language, pid, f"Split base solution into {len(chunks)} chunks")

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_text": source_text,
                "solution_text": solution_text,
                "chunks": chunks,
                "language": language,
            }, f, indent=2, ensure_ascii=False)
    else:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]
        _log(language, pid, f"Loaded {len(chunks)} existing chunks")

    _log(language, pid, f"Finished base solution. Total chunks: {len(chunks)}.")

    if len(chunks) > args.max_chunks:
        _log(language, pid, f"Too many chunks ({len(chunks)}). Skipping.")
        return

    # Build cumulative chunks
    cumulative_chunks = []
    current_cumulative = ""
    for chunk in chunks:
        current_cumulative += chunk + " "
        cumulative_chunks.append(current_cumulative.strip())

    # Process each chunk
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        if args.include_chunks and str(chunk_idx) not in args.include_chunks.split(","):
            continue

        _log(language, pid, f"START chunk {chunk_idx}/{len(chunks)-1}")

        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)

        solutions_file = chunk_dir / "solutions.json"
        existing_solutions = []
        valid_existing_solutions = []

        if solutions_file.exists() and not args.force:
            with open(solutions_file, "r", encoding="utf-8") as f:
                existing_solutions = json.load(f)

                # Recalculate accuracy if needed
                if not args.skip_recalculate:
                    updated = 0
                    for rollout in existing_solutions:
                        if "rollout" in rollout and "error" not in rollout:
                            if args.rollout_type == "forced_answer":
                                answer = extract_answer(rollout.get("full_cot", ""), problem, language)
                            else:
                                answer = extract_answer(rollout.get("rollout", ""), problem, language)
                            is_correct = check_answer_for_problem(answer, problem) if answer else False
                            if rollout.get("answer") != answer or rollout.get("is_correct") != is_correct:
                                rollout["answer"] = answer
                                rollout["is_correct"] = is_correct
                                updated += 1
                    if updated > 0:
                        with open(solutions_file, "w", encoding="utf-8") as f:
                            json.dump(existing_solutions, f, indent=2, ensure_ascii=False)

                valid_existing_solutions = [
                    s for s in existing_solutions
                    if "answer" in s and "error" not in s
                ]

        num_needed = args.num_rollouts - len(valid_existing_solutions)

        if num_needed > 0:
            _log(language, pid, f"START rollouts for chunk {chunk_idx} — generating {num_needed}")

            if args.provider == "Local":
                # Build prompts for batch generation
                prefix_without_chunk = full_prefix.replace(chunk, "").strip()
                prompts = []
                for _ in range(num_needed):
                    p = build_rollout_prompt(problem, prefix_without_chunk, language, args.rollout_type)
                    prompts.append(p)

                batch_results = generate_with_local_model_batch(
                    prompts, args.temperature, args.top_p, args.max_tokens
                )

                new_solutions = []
                for idx, result in enumerate(batch_results):
                    if "error" in result:
                        _log(language, pid, f"FINISH rollout {idx+1}/{num_needed} — resampling chunk {chunk_idx}; error={result['error']}")
                        new_solutions.append({"error": result["error"]})
                        continue

                    rollout_text = result.get("text", "")
                    chunks_split = split_solution_into_chunks(rollout_text, language)
                    chunk_resampled = chunks_split[0] if chunks_split else ""

                    full_cot = f"{prompts[idx]}{rollout_text}"
                    if args.rollout_type == "forced_answer":
                        answer = extract_answer(full_cot, problem, language)
                    else:
                        answer = extract_answer(rollout_text, problem, language)
                    is_correct = check_answer_for_problem(answer, problem) if answer else False

                    _log(language, pid, f"FINISH rollout {idx+1}/{num_needed} — resampling chunk {chunk_idx}; answer_correct={is_correct}")

                    new_solutions.append({
                        "chunk_removed": chunk,
                        "prefix_without_chunk": prefix_without_chunk,
                        "chunk_resampled": chunk_resampled,
                        "rollout": rollout_text,
                        "full_cot": full_cot,
                        "answer": answer,
                        "is_correct": is_correct,
                    })
            else:
                # API providers: generate in parallel
                tasks = [
                    generate_rollout(problem, chunk, full_prefix, language, args.temperature, args.rollout_type, chunk_idx=chunk_idx)
                    for _ in range(num_needed)
                ]
                new_solutions = await asyncio.gather(*tasks)
                new_solutions = list(new_solutions)

            all_solutions = existing_solutions + new_solutions
            with open(solutions_file, "w", encoding="utf-8") as f:
                json.dump(all_solutions, f, indent=2, ensure_ascii=False)

            _log(language, pid, f"Chunk {chunk_idx}: Saved {len(all_solutions)} solutions")
        else:
            _log(language, pid, f"Chunk {chunk_idx}: Already have {len(valid_existing_solutions)} valid solutions")

        _log(language, pid, f"END chunk {chunk_idx}/{len(chunks)-1}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    languages = [l.strip() for l in args.languages.split(",")]

    # Parse include_problems
    problem_ids = None
    if args.include_problems:
        problem_ids = [p.strip() for p in args.include_problems.split(",")]

    # Load problems
    print(f"Loading {args.dataset} problems for languages: {languages}")
    problems = load_dataset_problems(
        dataset=args.dataset,
        languages=languages,
        num_problems=args.num_problems,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        problem_ids=problem_ids,
    )

    if not problems:
        print("No problems loaded. Exiting.")
        return

    print(f"Loaded {len(problems)} problems available in all requested languages.")

    # Process each language
    for language in languages:
        output_dir = get_output_dir(language)
        print(f"\n{'='*60}")
        print(f"Processing language: {language} -> {output_dir}")
        print(f"{'='*60}")

        for problem_id, lang_problems in tqdm(problems.items(), desc=f"[{language}] Problems"):
            problem = lang_problems[language]
            await process_problem_language(problem, language, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
