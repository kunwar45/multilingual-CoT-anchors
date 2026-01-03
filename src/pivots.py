"""
Sentence-level pivot scoring utilities.

We approximate sentence "pivot" strength using the per-token logprob
gap between a reasoning-tuned model and a base model:

    Δ(t) = log p_reason(x_t | x_<t) - log p_base(x_t | x_<t)

For a sentence span, we aggregate |Δ(t)| over the tokens that fall
inside that span. High scores indicate locations where the reasoning
model's behavior diverges strongly from the base model.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .sentences import sentence_spans


def token_logprobs(
    model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log p(x_t | x_<t) for teacher-forced next-token prediction.

    Returns a tensor of shape [batch, T-1] aligned with target tokens
    input_ids[:, 1:].
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]  # predict next token
        logp = torch.log_softmax(logits, dim=-1)
        target = input_ids[:, 1:].unsqueeze(-1)
        token_logp = torch.gather(logp, dim=-1, index=target).squeeze(-1)
    return token_logp


def compute_token_gap(
    text: str,
    tok: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    reason_model: PreTrainedModel,
    device: torch.device,
    max_len: int = 2048,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Compute per-token logprob gap Δ(t) and corresponding character offsets.

    Returns:
        offsets: list of (start_char, end_char) for each *target* token
                 (i.e., tokens 1..T-1 in the input sequence)
        gap:     numpy array of shape [T-1] with Δ(t) values
    """
    if not text:
        return [], np.zeros(0, dtype=np.float32)

    enc = tok(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    # full offsets for all tokens, including the first one which has no target
    offsets_full = enc["offset_mapping"][0].tolist()

    base_lp = token_logprobs(base_model, input_ids, attn)[0].detach().cpu().numpy()
    reason_lp = token_logprobs(reason_model, input_ids, attn)[0].detach().cpu().numpy()

    gap = reason_lp - base_lp  # Δ logprob, length T-1
    # Align to target tokens: tokens 1..T-1 correspond to offsets_full[1:]
    offsets = offsets_full[1:]
    assert len(offsets) == len(gap)
    return offsets, gap


def sentence_pivot_scores(
    text: str,
    lang: str,
    tok: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    reason_model: PreTrainedModel,
    device: torch.device,
    max_len: int = 2048,
) -> pd.DataFrame:
    """
    Compute sentence-level pivot scores for a single reasoning trace.

    For each sentence, aggregate |Δ(t)| over tokens whose char offsets
    overlap the sentence span.
    """
    spans = sentence_spans(text, lang=lang)
    if not spans:
        return pd.DataFrame()

    token_offsets, gap = compute_token_gap(
        text, tok, base_model, reason_model, device, max_len=max_len
    )
    if len(gap) == 0:
        return pd.DataFrame()

    rows: List[dict] = []
    for si, (cs, ce, sent) in enumerate(spans):
        token_idxs = []
        for ti, (ts, te) in enumerate(token_offsets):
            # token overlaps sentence span
            if te <= cs or ts >= ce:
                continue
            token_idxs.append(ti)

        if not token_idxs:
            continue

        vals = np.abs(gap[token_idxs])
        rows.append(
            {
                "sent_idx": si,
                "char_start": cs,
                "char_end": ce,
                "n_tokens": len(token_idxs),
                "pivot_score": float(vals.mean()),
                "pivot_p95": float(np.percentile(vals, 95)),
                "sentence": sent,
            }
        )

    return pd.DataFrame(rows)


def detect_pivots(
    text: str,
    lang: str,
    tok: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    reason_model: PreTrainedModel,
    device: torch.device,
    max_len: int = 2048,
    top_k: int | None = None,
) -> Sequence[Tuple[int, float]]:
    """
    Convenience wrapper that returns a list of (sent_idx, pivot_score) pairs.

    If `top_k` is provided, only the top-k sentences by pivot_score are
    returned; otherwise all sentences with a defined score are returned.
    """
    df = sentence_pivot_scores(
        text=text,
        lang=lang,
        tok=tok,
        base_model=base_model,
        reason_model=reason_model,
        device=device,
        max_len=max_len,
    )
    if df.empty:
        return []
    df = df.sort_values("pivot_score", ascending=False)
    if top_k is not None:
        df = df.head(top_k)
    return list(zip(df["sent_idx"].tolist(), df["pivot_score"].tolist()))



