# coding=utf-8
"""
model/swift/modeling_qwen2.py
==============================
Port of SWIFT's model/swift/modeling_llama.py to Qwen2, following the SAME
subclass pattern SWIFT itself uses for Llama: subclass the real HF classes,
override ONLY what's needed for tree-attention masking + layer-skip logic.

Verified via direct inspection of modeling_llama.py (not assumed) before
writing this file:
  - Qwen2Attention: __init__ NOT overridden in the Llama version (goes
    straight to forward()) -- ported the same way here. forward() body is
    pure GQA attention math (num_heads/num_key_value_heads/head_dim/
    num_key_value_groups) that Qwen2Config exposes identically to
    LlamaConfig -- EXCEPT `self.config.pretraining_tp`, which does NOT
    exist on Qwen2Config (it's a Llama-specific tensor-parallelism-during-
    pretraining field). That branch is removed here rather than blindly
    copied -- see the `pretraining_tp` note in Qwen2Attention.forward().
  - Qwen2MLP: `pass` in the Llama version (zero overrides). Same here.
  - Qwen2DecoderLayer: skip-logic uses ONLY hidden_states, self.layer_id,
    and the module-level _attn_skip_layer_id_set/_mlp_skip_layer_id_set --
    confirmed architecture-agnostic. Copied near-verbatim, only class
    references swapped (Qwen2Attention/Qwen2MLP instead of Llama's).
  - Qwen2Model: _prepare_decoder_attention_mask (swift_mask splice) and
    forward() (draft_attn_skip_mask/draft_mlp_skip_mask passthrough) are
    both pure control flow, confirmed to reference nothing Llama-specific.
    Copied near-verbatim.
  - Qwen2ForCausalLM: only __init__ overridden in the Llama version, to
    swap in the custom *Model class. Same here.

apply_rotary_pos_emb / repeat_kv: confirmed Qwen2 does NOT define its own
versions (checked via hasattr on transformers.models.qwen2.modeling_qwen2)
-- both architectures share the same underlying utility functions, imported
from the Llama module below (this matches how HF itself organizes these
as shared RoPE/GQA utilities, not Llama-specific).
"""
import math
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as _Qwen2MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as _Qwen2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as _Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as _Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
# apply_rotary_pos_emb / repeat_kv: shared utilities, not Qwen2-specific.
# Reused from the Llama module (verified: Qwen2's own module does not
# define these -- see module docstring above).
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# Same causal-mask helpers SWIFT's modeling_llama.py defines locally.
from model.swift.modeling_llama import _make_causal_mask, _expand_mask

logger = logging.get_logger(__name__)

# Module-level globals -- SAME NAMES as modeling_llama.py's, but this is a
# SEPARATE module with its own globals (Python module-level state is per-
# module). inference_swift.py must import these from whichever modeling_*
# module matches the active --model_family, not mix the two.
enabled_draft = False
enabled_bitfit = False
_attn_skip_layer_id_set = []
_mlp_skip_layer_id_set = []

print('(Re-)Loading Qwen2 modeling...')


class Qwen2Attention(_Qwen2Attention):
    # NOT overriding __init__ -- verified the Llama version doesn't either;
    # Qwen2Attention.__init__ already correctly builds bias=True q/k/v
    # projections and bias=False o_proj from Qwen2Config, which is exactly
    # what we want (confirmed via direct __init__ inspection).

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        key_states = self.k_proj(hidden_states)
        query_states = self.q_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads (GQA -- same mechanism
        # Qwen2 and Llama both use, same repeat_kv utility)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # NOTE: the Llama version has a `self.config.pretraining_tp > 1`
        # branch here for tensor-parallel checkpoint splitting during
        # pretraining. Qwen2Config has NO pretraining_tp attribute --
        # verified, not assumed. Qwen2 was never pretrained with that TP
        # scheme, so this branch is correctly omitted rather than blindly
        # copied (copying it would throw AttributeError on every call).
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2MLP(_Qwen2MLP):
    # Zero overrides -- verified the Llama version is also a bare `pass`.
    pass


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.self_attn = Qwen2Attention(config=config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        draft_attn_skip_mask: torch.Tensor = None,
        draft_mlp_skip_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Skip-logic below is IDENTICAL to modeling_llama.py's
        # LlamaDecoderLayer.forward() -- confirmed architecture-agnostic
        # (uses only hidden_states, self.layer_id, and the module-level
        # skip-set globals; never touches attention/MLP internals directly).
        if self.training:
            if enabled_draft and draft_attn_skip_mask[self.layer_id].item():
                pass
            else:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states
            if enabled_draft and draft_mlp_skip_mask[self.layer_id].item():
                pass
            else:
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            if enabled_draft and self.layer_id in _attn_skip_layer_id_set:
                hidden_states = residual
                present_key_value = None
            else:
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states
            residual = hidden_states
            if enabled_draft and self.layer_id in _mlp_skip_layer_id_set:
                hidden_states = residual
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2Model(_Qwen2Model):
    """Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a Qwen2DecoderLayer."""

    def __init__(self, config: Qwen2Config):
        super(_Qwen2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = True
        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # (same as modeling_llama.py -- confirmed pure control flow, nothing
    # architecture-specific; the swift_mask splice mechanism is identical.)
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        if hasattr(self, "swift_mask") and self.swift_mask is not None and not enabled_draft:
            swift_mask = self.swift_mask
            swift_len = swift_mask.size(-1)
            combined_attention_mask[:, :, -swift_len:, -swift_len:][
                swift_mask == 0
            ] = combined_attention_mask.min()
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        draft_attn_skip_mask: torch.Tensor = None,
        draft_mlp_skip_mask: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            for past_key_value in past_key_values:
                if past_key_value is not None:
                    past_key_values_length = past_key_value[0].shape[2]
                    break
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                hidden_states.requires_grad_(True)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    draft_attn_skip_mask,
                    draft_mlp_skip_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    draft_attn_skip_mask=draft_attn_skip_mask,
                    draft_mlp_skip_mask=draft_mlp_skip_mask,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM(_Qwen2ForCausalLM):
    def __init__(self, config):
        super(_Qwen2ForCausalLM, self).__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        draft_attn_skip_mask: torch.Tensor = None,
        draft_mlp_skip_mask: torch.Tensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            draft_attn_skip_mask=draft_attn_skip_mask,
            draft_mlp_skip_mask=draft_mlp_skip_mask,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )


    def get_skip_layers(self):
        # Verified exact match against modeling_llama.py's LlamaForCausalLM
        # version -- trivial read of this module's own skip-set globals.
        return _attn_skip_layer_id_set, _mlp_skip_layer_id_set

    @contextmanager
    def self_draft(self, enabled=True, *args, **kwds):
        # Verified exact match against modeling_llama.py's LlamaForCausalLM
        # version. Toggles THIS module's enabled_draft global (module-level
        # state is per-module -- operates on modeling_qwen2.py's own
        # enabled_draft, not modeling_llama.py's).
        global enabled_draft
        enabled_draft = enabled
        try:
            yield None
        finally:
            enabled_draft = False

    def set_skip_layers(
        self, attn_skip_layer_id_set=None, mlp_skip_layer_id_set=None
    ):
        # Verified EXACT match against modeling_llama.py's LlamaForCausalLM
        # version (method inside the class, NOT a monkey-patched standalone
        # function -- corrected after an earlier wrong assumption). No
        # np.array() conversion; assigns through as-is. Either arg can be
        # updated independently (both default None).
        if attn_skip_layer_id_set is not None:
            global _attn_skip_layer_id_set
            _attn_skip_layer_id_set = attn_skip_layer_id_set

        if mlp_skip_layer_id_set is not None:
            global _mlp_skip_layer_id_set
            _mlp_skip_layer_id_set = mlp_skip_layer_id_set