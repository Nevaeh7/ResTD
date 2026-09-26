"""T5/mT5 retrieval models with training-only residual distillation heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from .backbone import LatentMT5 as LatentRMT5, LatentT5 as LatentRT5

from .losses import multi_horizon_restd_loss, capped_restd_lambda


class ResidualProjection(nn.Module):
    def __init__(self, hidden_size: int, residual_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, residual_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_state)


class ResTDTrainingMixin:
    """Shared implementation that leaves all inference methods untouched."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        restd_codebooks=None,
        **kwargs,
    ):
        """Restore external codebooks after Hugging Face initializes missing keys.

        Retriever checkpoints may omit the residual codebook buffers.
        Transformers initializes missing buffers after model construction,
        so external codebooks must be restored after pretrained loading.
        """

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            restd_codebooks=restd_codebooks,
            **kwargs,
        )
        if restd_codebooks is not None:
            actual = model[0] if isinstance(model, tuple) else model
            actual.load_restd_codebooks(restd_codebooks)
        return model

    def _init_restd(
        self,
        *,
        restd_codebooks: Sequence[torch.Tensor] | None,
        restd_max_horizon: int | None,
        restd_lambda: float | None,
        restd_teacher_temperature: float | None,
        restd_student_temperature: float | None,
        restd_horizon_decay: float | None,
        restd_collision_epsilon: float | None,
        restd_warmup_ratio: float | None,
    ) -> None:
        config = self.config
        if restd_codebooks is not None:
            codebooks = [tensor.detach().float().cpu() for tensor in restd_codebooks]
            if not codebooks:
                raise ValueError("ResTD requires at least one codebook")
            residual_size = int(codebooks[0].shape[1])
            if any(
                tensor.ndim != 2 or tensor.shape[1] != residual_size
                for tensor in codebooks
            ):
                raise ValueError("all ResTD codebooks must share one residual size")
            codebook_sizes = [int(tensor.shape[0]) for tensor in codebooks]
        else:
            residual_size = int(getattr(config, "restd_rq_dim", 0))
            codebook_sizes = list(getattr(config, "restd_codebook_sizes", []))
            if residual_size <= 0 or not codebook_sizes:
                raise ValueError(
                    "restd_codebooks are required when loading a non-ResTD checkpoint"
                )
            codebooks = [
                torch.zeros(size, residual_size, dtype=torch.float32)
                for size in codebook_sizes
            ]

        def configured(name: str, explicit, default):
            if explicit is not None:
                return explicit
            return getattr(config, name, default)

        max_horizon = int(configured("restd_max_horizon", restd_max_horizon, 4))
        if not 1 <= max_horizon <= len(codebooks):
            raise ValueError(f"restd_max_horizon must be in [1, {len(codebooks)}]")

        self.restd_enabled = True
        self.restd_lambda_max = float(configured("restd_lambda", restd_lambda, 0.1))
        self.restd_teacher_temperature = float(
            configured("restd_teacher_temperature", restd_teacher_temperature, 0.2)
        )
        self.restd_student_temperature = float(
            configured("restd_student_temperature", restd_student_temperature, 0.2)
        )
        self.restd_horizon_decay = float(
            configured("restd_horizon_decay", restd_horizon_decay, 0.7)
        )
        self.restd_collision_epsilon = float(
            configured("restd_collision_epsilon", restd_collision_epsilon, 0.1)
        )
        self.restd_warmup_ratio = float(
            configured("restd_warmup_ratio", restd_warmup_ratio, 0.1)
        )
        self.restd_adaptive_collision_margin = float(
            getattr(config, "restd_adaptive_collision_margin", 0.001)
        )
        self.restd_max_loss_ratio = float(getattr(config, "restd_max_loss_ratio", 0.05))
        config.restd_adaptive_collision_margin = self.restd_adaptive_collision_margin
        config.restd_max_loss_ratio = self.restd_max_loss_ratio
        self.restd_current_lambda = self.restd_lambda_max

        for level, codebook in enumerate(codebooks):
            self.register_buffer(f"restd_codebook_{level}", codebook, persistent=True)
        self.restd_num_codebooks = len(codebooks)
        self.restd_horizon_heads = nn.ModuleList(
            [
                ResidualProjection(self.config.d_model, residual_size)
                for _ in range(max_horizon)
            ]
        )
        self.restd_horizon_heads.apply(self._init_weights)

        config.restd_enabled = True
        config.restd_rq_dim = residual_size
        config.restd_num_codebooks = len(codebooks)
        config.restd_codebook_sizes = codebook_sizes
        config.restd_max_horizon = max_horizon
        config.restd_lambda = self.restd_lambda_max
        config.restd_teacher_temperature = self.restd_teacher_temperature
        config.restd_student_temperature = self.restd_student_temperature
        config.restd_horizon_decay = self.restd_horizon_decay
        config.restd_collision_epsilon = self.restd_collision_epsilon
        config.restd_warmup_ratio = self.restd_warmup_ratio
        config.architectures = [self.__class__.__name__]

        self._last_loss_restd = None
        self._last_loss_restd_weighted = None
        self._last_restd_metrics = {}

    def get_restd_codebooks(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"restd_codebook_{level}")
            for level in range(self.restd_num_codebooks)
        ]

    def load_restd_codebooks(self, codebooks: Sequence[torch.Tensor]) -> None:
        if len(codebooks) != self.restd_num_codebooks:
            raise ValueError("external codebook count does not match the ResTD model")
        with torch.no_grad():
            for level, source in enumerate(codebooks):
                target = getattr(self, f"restd_codebook_{level}")
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"external codebook {level} shape mismatch: "
                        f"{tuple(source.shape)} != {tuple(target.shape)}"
                    )
                source = source.detach().to(device=target.device, dtype=target.dtype)
                if not bool(torch.isfinite(source).all()):
                    raise ValueError(f"external codebook {level} is non-finite")
                target.copy_(source)

    def set_restd_progress(self, global_step: int, max_steps: int) -> float:
        warmup_steps = max(1, int(max_steps * self.restd_warmup_ratio))
        scale = min(1.0, max(0.0, float(global_step) / warmup_steps))
        self.restd_current_lambda = self.restd_lambda_max * scale
        return self.restd_current_lambda

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_outputs=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        category_labels=None,
        group_category_labels=None,
        final_category_ids=None,
        final_category_weights=None,
        final_category_mask=None,
        restd_residual_targets=None,
        restd_sid_targets=None,
        restd_valid_sid_mask=None,
        restd_item_idx=None,
        restd_enabled=None,
        **kwargs,
    ):
        """Consume ResTD-only inputs before dispatching to the T5 encoder.

        Residual tensors are training targets and must not be passed to the
        encoder as keyword arguments.
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
        is_inference_start = past_key_values is None or (
            isinstance(past_key_values, tuple) and len(past_key_values) == 0
        )
        if (
            not is_inference_start
            and hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() == 0
        ):
            is_inference_start = True

        if labels is not None:
            return self._forward_training(
                input_ids,
                attention_mask,
                decoder_input_ids,
                labels,
                category_labels,
                group_category_labels,
                encoder_outputs,
                return_dict,
                final_category_ids=final_category_ids,
                final_category_weights=final_category_weights,
                final_category_mask=final_category_mask,
                restd_residual_targets=restd_residual_targets,
                restd_sid_targets=restd_sid_targets,
                restd_valid_sid_mask=restd_valid_sid_mask,
                restd_item_idx=restd_item_idx,
                restd_enabled=restd_enabled,
            )
        if is_inference_start:
            return self._forward_inference_start(
                decoder_input_ids, encoder_outputs, attention_mask, return_dict
            )
        return self._forward_inference_step(
            decoder_input_ids,
            past_key_values,
            encoder_outputs,
            attention_mask,
            return_dict,
        )

    @torch.no_grad()
    def _sid_dynamics_metrics(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_sid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        length = self.restd_num_codebooks
        predictions = lm_logits[:, :length].argmax(dim=-1)
        targets = labels[:, :length]
        valid = valid_sid_mask[:, :length].bool() & targets.ne(-100)
        correct = predictions.eq(targets) & valid
        prefix_alive = torch.ones(
            predictions.shape[0], dtype=torch.bool, device=predictions.device
        )
        metrics = {}
        for level in range(length):
            level_valid = valid[:, level]
            denominator = level_valid.float().sum().clamp_min(1.0)
            metrics[f"sid_acc_l{level + 1}"] = (
                correct[:, level].float().sum() / denominator
            )
            prefix_alive = prefix_alive & (~level_valid | correct[:, level])
            metrics[f"prefix_survival_l{level + 1}"] = (
                prefix_alive[level_valid].float().mean()
                if bool(level_valid.any())
                else torch.zeros((), device=predictions.device)
            )
        return metrics

    def _forward_training(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        labels,
        category_labels,
        group_category_labels,
        encoder_outputs,
        return_dict,
        restd_residual_targets=None,
        restd_sid_targets=None,
        restd_valid_sid_mask=None,
        restd_item_idx=None,
        restd_enabled=None,
        final_category_ids=None,
        final_category_weights=None,
        final_category_mask=None,
        **kwargs,
    ):
        del (
            input_ids,
            restd_item_idx,
            final_category_ids,
            final_category_weights,
            final_category_mask,
            kwargs,
        )
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_inputs_embeds = self.shared(decoder_input_ids)
        current_input_embeds = decoder_inputs_embeds[:, 0:1, :]
        past_key_values = None
        total_category_loss = torch.zeros((), device=decoder_input_ids.device)
        total_contrastive_loss = torch.zeros((), device=decoder_input_ids.device)

        for step in range(self.num_latent_steps):
            outputs = self.decoder(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            current_latent_state = last_hidden_state.squeeze(1)
            if category_labels is not None:
                step_logits, step_query_emb = self.category_heads[step](
                    current_latent_state
                )
                if category_labels[:, step].ne(-100).any():
                    total_category_loss = total_category_loss + self.loss_fct(
                        step_logits, category_labels[:, step]
                    )
                if group_category_labels is not None:
                    step_group_ids = group_category_labels[:, step, :]
                    valid_group = step_group_ids.ne(-100)
                    safe_group_ids = step_group_ids.masked_fill(~valid_group, 0)
                    step_group_embs = F.embedding(
                        safe_group_ids,
                        self.category_heads[step].classifier.weight,
                    )
                    query_norm = F.normalize(step_query_emb, dim=-1)
                    group_norm = F.normalize(step_group_embs, dim=-1)
                    batch, group, dim = group_norm.shape
                    logits = query_norm @ group_norm.view(-1, dim).t() / 0.1
                    block = torch.eye(
                        batch, device=logits.device, dtype=torch.bool
                    ).repeat_interleave(group, dim=1)
                    positive = block & valid_group.view(-1).unsqueeze(0)
                    selected = F.log_softmax(logits, dim=1).masked_select(positive)
                    if selected.numel() > 0:
                        total_contrastive_loss = (
                            total_contrastive_loss - selected.mean()
                        )
            current_input_embeds = last_hidden_state

        first_sid_hidden = current_input_embeds
        scaled_first = first_sid_hidden
        if self.config.tie_word_embeddings:
            scaled_first = scaled_first * (self.model_dim**-0.5)
        logits_first_token = self.lm_head(scaled_first)

        text_inputs_embeds = decoder_inputs_embeds[:, 1:, :]
        remaining_sid_hidden = None
        lm_logits_rest = None
        if text_inputs_embeds.shape[1] > 0:
            outputs = self.decoder(
                inputs_embeds=text_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            remaining_sid_hidden = outputs.last_hidden_state
            scaled_remaining = remaining_sid_hidden
            if self.config.tie_word_embeddings:
                scaled_remaining = scaled_remaining * (self.model_dim**-0.5)
            lm_logits_rest = self.lm_head(scaled_remaining)

        if lm_logits_rest is None:
            lm_logits = logits_first_token
            sid_hidden_states = first_sid_hidden
        else:
            lm_logits = torch.cat([logits_first_token, lm_logits_rest], dim=1)
            sid_hidden_states = torch.cat(
                [first_sid_hidden, remaining_sid_hidden], dim=1
            )

        lm_loss = self.loss_fct(
            lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1)
        )
        final_loss = lm_loss
        if category_labels is not None:
            final_loss = final_loss + self.alpha * total_category_loss
        if group_category_labels is not None:
            final_loss = final_loss + self.beta * total_contrastive_loss

        use_restd = self.restd_enabled if restd_enabled is None else restd_enabled
        restd_loss = sid_hidden_states.sum() * 0.0
        restd_metrics = {}
        applied_lambda = torch.zeros((), device=lm_loss.device)
        if use_restd:
            if restd_residual_targets is None or restd_sid_targets is None:
                raise ValueError(
                    "ResTD is enabled but residual/SID targets were not provided"
                )
            if restd_valid_sid_mask is None:
                restd_valid_sid_mask = torch.ones_like(
                    restd_sid_targets, dtype=torch.bool
                )
            restd_loss, restd_metrics = multi_horizon_restd_loss(
                decoder_hidden_states=sid_hidden_states,
                residual_targets=restd_residual_targets,
                codebooks=self.get_restd_codebooks(),
                horizon_heads=self.restd_horizon_heads,
                teacher_temperature=self.restd_teacher_temperature,
                student_temperature=self.restd_student_temperature,
                horizon_decay=self.restd_horizon_decay,
                valid_sid_mask=restd_valid_sid_mask,
                final_sid_targets=restd_sid_targets,
                collision_epsilon=self.restd_collision_epsilon,
                adaptive_collision_margin=self.restd_adaptive_collision_margin,
            )
            applied_lambda = capped_restd_lambda(
                lm_loss,
                restd_loss,
                scheduled_lambda=self.restd_current_lambda,
                max_loss_ratio=self.restd_max_loss_ratio,
            )
            final_loss = final_loss + applied_lambda * restd_loss
            restd_metrics.update(
                self._sid_dynamics_metrics(lm_logits, labels, restd_valid_sid_mask)
            )

        self._last_loss_lm = lm_loss
        self._last_loss_category = self.alpha * total_category_loss
        self._last_loss_contrastive = self.beta * total_contrastive_loss
        self._last_loss_restd = restd_loss
        self._last_loss_restd_weighted = applied_lambda * restd_loss
        self._last_restd_metrics = restd_metrics

        if not return_dict:
            return final_loss, lm_logits
        output = Seq2SeqLMOutput(
            loss=final_loss,
            logits=lm_logits,
            past_key_values=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )
        output.loss_lm = lm_loss
        output.loss_category = self.alpha * total_category_loss
        output.loss_contrastive = self.beta * total_contrastive_loss
        output.loss_restd = restd_loss
        output.loss_restd_weighted = applied_lambda * restd_loss
        output.restd_metrics = restd_metrics
        output.sid_hidden_states = sid_hidden_states
        return output


class ResTDT5(ResTDTrainingMixin, LatentRT5):
    def __init__(
        self,
        config,
        latent_token_length=None,
        num_categories_list=None,
        alpha=None,
        beta=None,
        restd_codebooks=None,
        restd_max_horizon=None,
        restd_lambda=None,
        restd_teacher_temperature=None,
        restd_student_temperature=None,
        restd_horizon_decay=None,
        restd_collision_epsilon=None,
        restd_warmup_ratio=None,
    ):
        super().__init__(
            config,
            latent_token_length=latent_token_length,
            num_categories_list=num_categories_list,
            alpha=alpha,
            beta=beta,
        )
        self._init_restd(
            restd_codebooks=restd_codebooks,
            restd_max_horizon=restd_max_horizon,
            restd_lambda=restd_lambda,
            restd_teacher_temperature=restd_teacher_temperature,
            restd_student_temperature=restd_student_temperature,
            restd_horizon_decay=restd_horizon_decay,
            restd_collision_epsilon=restd_collision_epsilon,
            restd_warmup_ratio=restd_warmup_ratio,
        )


class ResTDMT5(ResTDTrainingMixin, LatentRMT5):
    def __init__(
        self,
        config,
        latent_token_length=None,
        num_categories_list=None,
        alpha=None,
        beta=None,
        restd_codebooks=None,
        restd_max_horizon=None,
        restd_lambda=None,
        restd_teacher_temperature=None,
        restd_student_temperature=None,
        restd_horizon_decay=None,
        restd_collision_epsilon=None,
        restd_warmup_ratio=None,
    ):
        super().__init__(
            config,
            latent_token_length=latent_token_length,
            num_categories_list=num_categories_list,
            alpha=alpha,
            beta=beta,
        )
        self._init_restd(
            restd_codebooks=restd_codebooks,
            restd_max_horizon=restd_max_horizon,
            restd_lambda=restd_lambda,
            restd_teacher_temperature=restd_teacher_temperature,
            restd_student_temperature=restd_student_temperature,
            restd_horizon_decay=restd_horizon_decay,
            restd_collision_epsilon=restd_collision_epsilon,
            restd_warmup_ratio=restd_warmup_ratio,
        )


class GenerationOnlyMixin:
    """Load a converged native retriever without adding training-only parameters."""

    forward = ResTDTrainingMixin.forward


class RetrievalT5(GenerationOnlyMixin, LatentRT5):
    pass


class RetrievalMT5(GenerationOnlyMixin, LatentRMT5):
    pass
