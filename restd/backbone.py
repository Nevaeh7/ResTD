"""Latent category reasoning for T5 and mT5 retrieval models."""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class CategoryClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedding = self.projector(x)
        logits = self.classifier(embedding)
        return logits, embedding


class LatentReasoningMixin:
    def __init__(
        self,
        config,
        latent_token_length=None,
        num_categories_list=None,
        alpha=None,
        beta=None,
    ):
        super().__init__(config)

        latent_token_length = latent_token_length or getattr(
            config, "latent_token_length", 3
        )
        num_categories_list = num_categories_list or getattr(
            config, "num_categories_list", [10, 50, 100]
        )
        alpha = getattr(config, "alpha", 0.1) if alpha is None else alpha
        beta = getattr(config, "beta", 0.05) if beta is None else beta
        config.latent_token_length = latent_token_length
        config.num_categories_list = list(num_categories_list)
        config.alpha, config.beta = alpha, beta
        self.num_latent_steps = latent_token_length
        num_categories_list = num_categories_list[:latent_token_length]

        self.category_heads = nn.ModuleList(
            [CategoryClassifier(config.d_model, num_c) for num_c in num_categories_list]
        )

        self.alpha = alpha
        self.beta = beta

        self.category_heads.apply(self._init_weights)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        self._last_loss_lm = None
        self._last_loss_category = None
        self._last_loss_contrastive = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_inference_start(
        self, decoder_input_ids, encoder_outputs, attention_mask, return_dict
    ):
        current_input_embeds = self.shared(decoder_input_ids)
        past_key_values = None
        for _ in range(self.num_latent_steps):
            outputs = self.decoder(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            current_input_embeds = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        sequence_output = current_input_embeds
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        if not return_dict:
            return (lm_logits, past_key_values)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def _forward_inference_step(
        self,
        decoder_input_ids,
        past_key_values,
        encoder_outputs,
        attention_mask,
        return_dict,
    ):
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits, outputs.past_key_values)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def predict_latent_categories(self, input_ids, attention_mask=None, top_k=100):
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        decoder_input_ids = torch.full(
            (input_ids.shape[0], 1),
            self.config.decoder_start_token_id,
            device=input_ids.device,
            dtype=torch.long,
        )

        current_input_embeds = self.shared(decoder_input_ids)
        past_key_values = None

        all_topk_ids = []

        for i in range(self.num_latent_steps):
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
            current_input_embeds = last_hidden_state

            current_state = last_hidden_state.squeeze(1)
            logits, _ = self.category_heads[i](current_state)

            _, topk_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            all_topk_ids.append(topk_indices)

        return all_topk_ids


class LatentT5(LatentReasoningMixin, T5ForConditionalGeneration):
    pass


class LatentMT5(LatentReasoningMixin, MT5ForConditionalGeneration):
    pass
