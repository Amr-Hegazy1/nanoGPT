"""
Run lm-evaluation-harness against nanoGPT checkpoints.

Example:
    source .venv/bin/activate
    python scripts/lm_eval_nanogpt.py \
        --checkpoint out/ckpt.pt \
        --tasks hellaswag,arc_easy \
        --device cuda \
        --dtype bfloat16

Core responsibilities:
- Load a nanoGPT checkpoint (including oracle state) and expose it as an lm-eval LM.
- Provide GPT-2 BPE or character-level tokenization depending on dataset meta.pkl.
- Implement loglikelihood, rolling perplexity, and generate-until hooks expected by lm-eval.
- Keep everything self-contained in this repo; no edits to lm-eval-harness.

Arguments (CLI):
- --checkpoint: Path to ckpt.pt or its directory (required).
- --tasks: Comma-separated lm-eval task names (required).
- --device: Target device string (cuda/cpu/etc.).
- --dtype: Numeric precision (auto|float32|float16|bfloat16).
- --recurrent_depth: Optional override for recurrent shared depth at eval time.
- --max_gen_toks: Cap on generation length for generative tasks.
- --meta_path: Optional explicit meta.pkl for tokenizer info.
- --output_path: Where to store full JSON results.
- --compile: Enable torch.compile on the loaded model.
- --seed: Seed for sampling paths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make the local lm-evaluation-harness importable without installing it.
REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_PATH = REPO_ROOT / "lm-evaluation-harness"
if str(HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HARNESS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm_eval import simple_evaluate  # type: ignore  # noqa: E402
from lm_eval.api.model import TemplateLM  # type: ignore  # noqa: E402
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window  # type: ignore  # noqa: E402
import tiktoken  # type: ignore  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Normalize string/torch dtype spec to a torch dtype.

    Args:
        dtype: String alias or torch dtype.
    Returns:
        torch.dtype resolved from the input.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _load_checkpoint(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[GPT, dict]:
    """Load a ckpt.pt produced by train.py, strip DDP prefixes, move to device/dtype.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Target torch.device for model weights.
        dtype: Target dtype for parameters.
    Returns:
        (model, checkpoint_dict) with oracle state restored if present.
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    model.load_oracle_state(checkpoint.get("oracle_state"))
    model.eval()
    model.to(device=device, dtype=dtype)
    return model, checkpoint


def _build_tokenizer(meta_path: Optional[Path], dataset_name: Optional[str]):
    """Return encode/decode functions, eot token id, tokenizer name, and vocab size.

    Prefers dataset-specific meta.pkl when available; otherwise GPT-2 BPE via tiktoken.

    Args:
        meta_path: Optional explicit path to meta.pkl.
        dataset_name: Name hint from checkpoint config.
    Returns:
        (encode_fn, decode_fn, eot_token_id, tokenizer_name, vocab_size)
    """
    if meta_path is not None and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi = meta["stoi"]
        itos = meta["itos"]
        vocab_size = meta.get("vocab_size", len(stoi))

        def encode(s: str, add_special_tokens: bool | None = None) -> list[int]:
            return [stoi[c] for c in s]

        def decode(tokens: Iterable[int]) -> str:
            return "".join(itos[int(t)] for t in tokens)

        eot = stoi.get("<|endoftext|>", 0)
        name = f"nanogpt-meta-{dataset_name or meta_path.parent.name}"
        return encode, decode, eot, name, vocab_size

    enc = tiktoken.get_encoding("gpt2")

    def encode(s: str, add_special_tokens: bool | None = None) -> list[int]:
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(tokens: Iterable[int]) -> str:
        return enc.decode(list(tokens))

    return encode, decode, enc.eot_token, "tiktoken-gpt2", enc.n_vocab


class NanoGPTLM(TemplateLM):
    """lm-eval LM wrapper around our nanoGPT model.

    Args:
        checkpoint_path: Path to ckpt.pt or its containing dir.
        device: Device string (cuda/cpu/etc.).
        dtype: Precision string/torch dtype.
        max_gen_toks: Max new tokens for generation.
        recurrent_depth: Optional override for recurrent shared weights depth.
        meta_path: Optional explicit tokenizer meta.pkl path.
        compile_model: Whether to torch.compile the loaded model.
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        dtype: str | torch.dtype = "auto",
        max_gen_toks: int = 256,
        recurrent_depth: Optional[int] = None,
        meta_path: Optional[str] = None,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick device
        self.dtype = _resolve_dtype(dtype)  # normalize dtype choice
        self.max_gen_toks = max_gen_toks  # store generation cap

        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "ckpt.pt"  # default filename inside dir
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        self.model, checkpoint = _load_checkpoint(ckpt_path, self.device, self.dtype)  # load weights
        if compile_model:
            self.model = torch.compile(self.model)  # type: ignore[arg-type]  # optional compile for speed
        config = checkpoint.get("config", {})  # training config for dataset hint
        dataset_name = config.get("dataset") if isinstance(config, dict) else None
        resolved_meta = Path(meta_path) if meta_path else None  # user override
        if resolved_meta is None and dataset_name:
            candidate = REPO_ROOT / "data" / dataset_name / "meta.pkl"  # expected location
            resolved_meta = candidate if candidate.exists() else None

        self.encode, self.decode, self._eot_token_id, tok_name, vocab_size = _build_tokenizer(resolved_meta, dataset_name)
        self._tokenizer_name = f"{tok_name}-v{vocab_size}"  # used for cache fingerprinting
        self.max_length = int(self.model.config.block_size)  # context window
        self.vocab_size = vocab_size  # tokenizer vocab size
        self.recurrent_depth = (
            recurrent_depth if recurrent_depth is not None else getattr(self.model.config, "recurrent_depth", None)
        )  # optional depth override
        # TemplateLM expects batch_size; we keep it at 1 because the model lacks padding support.
        self.batch_size = 1
        self._default_gen_kwargs = {"temperature": 1.0, "top_k": None, "top_p": None}  # defaults for generation

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, **_: object) -> list[int]:
        return self.encode(string, add_special_tokens)

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.decode(tokens)

    def _autocast(self):
        """Autocast only when on GPU and using reduced precision.

        Returns:
            Context manager for autocast or no-op.
        """
        if self.device.type == "cpu" or self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def _score_tokens(self, context_enc: list[int], continuation_enc: list[int]) -> tuple[float, bool]:
        """Compute total logprob and greedy-match flag for a continuation.

        Args:
            context_enc: Tokenized context prefix.
            continuation_enc: Tokenized continuation to score.
        Returns:
            (total_logprob, is_greedy_match)
        """
        if len(continuation_enc) == 0:
            return 0.0, True
        if len(continuation_enc) > self.max_length:
            raise ValueError(
                f"Continuation is longer than model context window ({len(continuation_enc)} > {self.max_length})."
            )
        combined = context_enc + continuation_enc  # concat for a single forward
        if len(combined) > self.max_length + 1:
            combined = combined[-(self.max_length + 1) :]  # left-truncate to fit
        input_tokens = combined[:-1]  # tokens fed to model
        target_tokens = combined[1:]  # tokens predicted

        idx = torch.tensor(input_tokens, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, T]
        targets = torch.tensor(target_tokens, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, T]
        with torch.inference_mode():
            with self._autocast():
                # Pass targets to obtain logits for every position; loss is ignored.
                logits, _, _ = self.model(idx, targets=targets, n=self.recurrent_depth)
        log_probs = F.log_softmax(logits, dim=-1)  # convert logits to logprobs

        cont_len = len(continuation_enc)
        cont_logits = log_probs[:, -cont_len:, :]  # slice continuation positions
        cont_target = (
            torch.tensor(continuation_enc[-cont_len:], device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # shape [1, L, 1]
        token_logprobs = torch.gather(cont_logits, 2, cont_target).squeeze(-1)  # pick correct token logprobs
        total_logprob = float(token_logprobs.sum().item())  # sum over continuation
        greedy = bool((cont_logits.argmax(dim=-1) == cont_target.squeeze(-1)).all().item())  # greedy match?
        return total_logprob, greedy

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **_: object,
    ) -> list[tuple[float, bool]]:
        """lm-eval hook: batch scoring wrapper with progress bar."""
        results: list[tuple[float, bool]] = []
        iterator = requests
        if not disable_tqdm:
            iterator = tqdm(iterator, desc="loglikelihood", disable=False)
        for (_, context_enc, continuation_enc) in iterator:
            results.append(self._score_tokens(context_enc, continuation_enc))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        """lm-eval hook: full-text logprob using rolling windows."""
        scores: list[float] = []
        iterator = requests
        if not disable_tqdm:
            iterator = tqdm(iterator, desc="loglikelihood_rolling", disable=False)
        for (text,) in iterator:
            token_list = self.tok_encode(text)
            if not token_list:
                scores.append(0.0)
                continue
            logprob_total = 0.0
            windows = map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=token_list,
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )
            for context_enc, continuation_enc in windows:
                logprob, _ = self._score_tokens(context_enc, continuation_enc)
                logprob_total += logprob
            scores.append(logprob_total)
        return scores

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Greedy/temperature sampling loop with optional top-k/top-p."""
        idx = input_ids  # running sequence
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_length else idx[:, -self.max_length :]  # respect context window
            logits, _, _ = self.model(idx_cond, n=self.recurrent_depth)  # forward
            logits = logits[:, -1, :]  # last token logits
            if temperature != 1.0:
                logits = logits / temperature  # scale temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # top-k filter
                logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, float("-inf")), logits)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # sort for nucleus
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 0] = False  # always keep the first token
                sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)  # restore order
            probs = F.softmax(logits, dim=-1)  # convert to probs
            next_token = torch.multinomial(probs, num_samples=1)  # sample token
            idx = torch.cat((idx, next_token), dim=1)  # append
        return idx

    def _apply_until(self, generated_tokens: list[int], until: Optional[list[str]]) -> str:
        text = self.tok_decode(generated_tokens)
        if not until:
            return text
        stop_positions = [pos for u in until if (pos := text.find(u)) != -1]
        if stop_positions:
            text = text[: min(stop_positions)]
        return text

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """lm-eval hook: generate continuations until stop sequences or length."""
        results: list[str] = []
        iterator = requests
        if not disable_tqdm:
            iterator = tqdm(iterator, desc="generate_until", disable=False)
        for context, gen_kwargs in iterator:
            kwargs = {**self._default_gen_kwargs, **(gen_kwargs or {})}
            max_gen_toks = int(kwargs.pop("max_gen_toks", self.max_gen_toks))
            until = kwargs.pop("until", None)
            until = [until] if isinstance(until, str) else until
            temperature = float(kwargs.pop("temperature", 0.5))
            top_k = kwargs.pop("top_k", None)
            top_p = kwargs.pop("top_p", None)

            context_tokens = self.tok_encode(context)  # encode prompt
            max_ctx_len = max(1, self.max_length - max_gen_toks)  # leave room to generate
            if len(context_tokens) > max_ctx_len:
                context_tokens = context_tokens[-max_ctx_len:]  # truncate left if needed
            idx = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, T]
            with torch.inference_mode():
                generated = self._generate_tokens(
                    idx,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            new_tokens = generated[0].tolist()[len(context_tokens) :]
            results.append(self._apply_until(new_tokens, until))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nanoGPT checkpoints with lm-evaluation-harness.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt or its containing directory.")
    parser.add_argument("--tasks", required=True, help="Comma-separated list of lm-eval task names.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task example cap.")
    parser.add_argument("--device", default=None, help="Device string, e.g., cuda or cpu.")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto|float32|float16|bfloat16.")
    parser.add_argument("--recurrent_depth", type=int, default=None, help="Override recurrent depth during eval.")
    parser.add_argument("--max_gen_toks", type=int, default=256, help="Max new tokens for generative tasks.")
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to meta.pkl for tokenization.")
    parser.add_argument("--output_path", type=str, default=None, help="Where to write the full results JSON.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the loaded model.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lm = NanoGPTLM(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        max_gen_toks=args.max_gen_toks,
        recurrent_depth=args.recurrent_depth,
        meta_path=args.meta_path,
        compile_model=args.compile,
    )

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=lm.batch_size,
        device=args.device,
        limit=args.limit,
        gen_kwargs={"max_gen_toks": args.max_gen_toks},
    )

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote full results to {out_path}")

    print(json.dumps(results.get("results", results), indent=2))


if __name__ == "__main__":
    main()
