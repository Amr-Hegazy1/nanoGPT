"""
Oracle teacher utilities that supervise the learned stopping controller.

The teacher is just another GPT instance that runs with frozen weights and
collects the depth at which each token (or sequence) becomes correct.
Those annotations are later used by the stopping strategies to provide
targets for depth-aware losses.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class OracleAnnotations:
    """Lightweight container with the per-token / per-sequence depth targets."""

    token_stop_depths: torch.Tensor  # (B, T)
    token_difficulty: torch.Tensor  # (B, T)
    token_stop_targets: torch.Tensor  # (B, T, L)
    token_confidence: torch.Tensor  # (B, T)
    sequence_stop_depths: torch.Tensor  # (B,)
    sequence_difficulty: torch.Tensor  # (B,)
    sequence_stop_targets: torch.Tensor  # (B, L)
    sequence_confidence: torch.Tensor  # (B,)
    valid_mask: torch.Tensor  # (B, T)
    num_layers: int


class OracleTeacher:
    """
    Wraps a frozen reference model and produces oracle annotations that describe
    how many recurrent layers it takes to predict each token correctly.

    The teacher itself is just another GPT configured with the same architecture
    as the student but with gradients disabled. During training, we repeatedly
    run the shared block up to the requested depth, snapshot the logits, and
    convert them into targets such as "stop at layer k" for every token.
    """

    def __init__(self, model: nn.Module, max_depth: Optional[int] = None) -> None:
        self.model = model
        self.max_depth = max_depth
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[oracle] warning: missing keys {missing} unexpected keys {unexpected}")
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[oracle] warning: missing keys {missing} unexpected keys {unexpected}")
        self.model.eval()

    def annotate(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        requested_depth: int,
        tokenwise: bool,
    ) -> OracleAnnotations:
        """
        Run the frozen teacher for `requested_depth` recurrent steps and record
        when each token or sequence first becomes correct.

        Args:
            idx: token ids (B, T) fed into the teacher.
            targets: supervision labels aligned with idx (B, T).
            requested_depth: how many shared steps to expand.
            tokenwise: currently unused hook for per-token heuristics.

        Returns:
            OracleAnnotations with depth targets, difficulty scores, and masks.
        """
        _ = tokenwise  # reserved for future per-token heuristics
        if targets is None:
            raise ValueError("Oracle annotations require supervision targets.")
        if targets.device != idx.device:
            raise ValueError("Targets and inputs must be on the same device for oracle annotations.")

        depth = self._resolve_depth(requested_depth)
        with torch.no_grad():
            logits_per_layer = self._collect_logits(idx, depth)

        num_layers = len(logits_per_layer)
        if num_layers == 0:
            raise RuntimeError("Oracle teacher did not produce any layers.")

        device = idx.device
        if targets.device != device:
            targets = targets.to(device, non_blocking=True)
        valid_mask = targets.ne(-1)
        fallback_depth = torch.tensor(num_layers - 1, device=device, dtype=torch.long)
        token_stop_depths = torch.full_like(targets, fill_value=-1, dtype=torch.long, device=device)
        token_confidence = torch.zeros_like(targets, dtype=torch.float32, device=device)
        fallback_confidence = torch.zeros_like(targets, dtype=torch.float32, device=device)

        # Scan each depth until we find when a token first becomes correct.
        for depth_idx, logits in enumerate(logits_per_layer):
            log_probs = torch.log_softmax(logits, dim=-1)
            target_logp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            target_prob = target_logp.exp().to(token_confidence.dtype)
            fallback_confidence = target_prob
            preds = torch.argmax(logits, dim=-1)
            correct = preds.eq(targets) & valid_mask
            needs_update = (token_stop_depths < 0) & correct
            token_stop_depths = torch.where(needs_update, torch.full_like(token_stop_depths, depth_idx), token_stop_depths)
            token_confidence = torch.where(needs_update, target_prob, token_confidence)

        token_stop_depths = torch.where(
            valid_mask,
            torch.where(token_stop_depths >= 0, token_stop_depths, fallback_depth),
            torch.zeros_like(token_stop_depths),
        )
        token_confidence = torch.where(
            valid_mask,
            torch.where(token_stop_depths >= 0, token_confidence, fallback_confidence),
            torch.zeros_like(token_confidence),
        )

        denom = float(num_layers)
        token_difficulty = (token_stop_depths.to(torch.float32) + 1.0) / denom
        token_difficulty = torch.where(valid_mask, token_difficulty, torch.zeros_like(token_difficulty))

        layer_ids = torch.arange(num_layers, device=device)
        expanded_depths = token_stop_depths.unsqueeze(-1).expand(-1, -1, num_layers)
        # Build one-hot targets of shape (B, T, num_layers).
        token_stop_targets = torch.where(
            expanded_depths.eq(layer_ids.view(1, 1, -1)),
            torch.ones_like(expanded_depths, dtype=torch.float32),
            torch.zeros_like(expanded_depths, dtype=torch.float32),
        )
        token_stop_targets = token_stop_targets * valid_mask.unsqueeze(-1)

        any_valid = valid_mask.any(dim=1)
        sequence_stop_depths = torch.where(
            any_valid,
            token_stop_depths.masked_fill(~valid_mask, 0).max(dim=1).values,
            torch.zeros(valid_mask.size(0), device=device, dtype=torch.long),
        )
        sequence_difficulty = (sequence_stop_depths.to(torch.float32) + 1.0) / denom
        sequence_difficulty = torch.where(any_valid, sequence_difficulty, torch.zeros_like(sequence_difficulty))
        sequence_depth_mask = token_stop_depths.eq(sequence_stop_depths.unsqueeze(-1)) & valid_mask
        sequence_confidence = torch.where(
            any_valid,
            torch.where(sequence_depth_mask, token_confidence, torch.zeros_like(token_confidence)).max(dim=1).values,
            torch.zeros_like(sequence_stop_depths, dtype=torch.float32),
        )

        seq_depths_expanded = sequence_stop_depths.unsqueeze(-1).expand(-1, num_layers)
        # Sequence-level one-hot targets (B, num_layers).
        sequence_stop_targets = torch.where(
            seq_depths_expanded.eq(layer_ids.view(1, -1)),
            torch.ones_like(seq_depths_expanded, dtype=torch.float32),
            torch.zeros_like(seq_depths_expanded, dtype=torch.float32),
        )
        sequence_stop_targets = sequence_stop_targets * any_valid.unsqueeze(-1)

        return OracleAnnotations(
            token_stop_depths=token_stop_depths,
            token_difficulty=token_difficulty,
            token_stop_targets=token_stop_targets,
            token_confidence=token_confidence,
            sequence_stop_depths=sequence_stop_depths,
            sequence_difficulty=sequence_difficulty,
            sequence_stop_targets=sequence_stop_targets,
            sequence_confidence=sequence_confidence,
            valid_mask=valid_mask,
            num_layers=num_layers,
        )

    def _resolve_depth(self, requested_depth: int) -> int:
        teacher_depth = self._teacher_depth()
        depth = min(requested_depth, teacher_depth)
        if self.max_depth is not None:
            depth = min(depth, self.max_depth)
        return depth

    def _teacher_depth(self) -> int:
        cfg = self.model.config
        if cfg.share_parameters_across_layers and cfg.recurrent_shared_weights:
            return cfg.recurrent_depth
        return cfg.n_layer

    def _collect_logits(self, idx: torch.Tensor, depth: int) -> List[torch.Tensor]:
        model = self.model
        if idx.device != self.device:
            idx = idx.to(self.device, non_blocking=True)
        device = idx.device
        x = model.transformer.wte(idx)
        positions = torch.arange(0, idx.size(1), dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(positions)
        x = model.transformer.drop(x + pos_emb)

        states = []
        if model.config.share_parameters_across_layers:
            shared_blocks = model.transformer.h
            if len(shared_blocks) == 0:
                raise RuntimeError("No shared blocks available for recurrent forwarding.")
            prelude_output = None
            if model.fixed_edge_blocks and model.fixed_head is not None:
                for block in model.fixed_head:
                    x = block(x)
                if model.recurrent_prelude_injection:
                    prelude_output = x.clone()

            def shared_block_fn(tensor):
                out = tensor
                for block_idx, block in enumerate(shared_blocks):
                    inject_prelude = model.recurrent_prelude_injection and (
                        not getattr(model, 'recurrent_prelude_injection_first_block_only', False) or block_idx == 0
                    )
                    if inject_prelude:
                        out = model._forward_shared_block(block, out, prelude_output=prelude_output)
                    else:
                        out = model._forward_shared_block(block, out)
                return out

            for _ in range(depth):
                x = shared_block_fn(x)
                states.append(x)
        else:
            blocks = model.transformer.h
            capped_depth = min(depth, len(blocks))
            for i in range(capped_depth):
                x = blocks[i](x)
                states.append(x)

        logits_per_layer = []
        for hidden in states:
            head_in = hidden
            if model.fixed_edge_blocks and model.fixed_tail is not None:
                for block in model.fixed_tail:
                    head_in = block(head_in)
            head_in = model.transformer.ln_f(head_in)
            logits = model.lm_head(head_in)
            logits_per_layer.append(logits)
        return logits_per_layer
