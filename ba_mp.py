import torch


from transformers.cache_utils import DynamicCache, Cache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import AutoConfig, GenerationConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


from typing import List, Union, Dict, Mapping, Optional, Tuple, TypedDict
import torch
from torch import Tensor
import os
import json
import numpy as np
from functools import partial
from contextlib import nullcontext
# from transformers import AutoModel, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
# from transformers.modeling_utils import PreTrainedModel
# from transformers.models.auto import AutoTokenizer
# from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
from transformers import MistralModel, MistralConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from einops import rearrange, repeat
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import types
import gc
import time
logger = logging.get_logger(__name__)


# tool functions to calculate RoPE

def rotate_half(x):
    # tool function in apply_rotart_pos_emb
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    ## apply RoPE to K
    # the code is from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    # the q_embed is commented, and return onlt K pos, for KV cache calculation
    # print(cos.size())
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # print(cos.size())
    # print(k.size())
    # print(rotate_half(k).size())
    #q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    #eturn q_embed, k_embed
    return k_embed.to(dtype=torch.bfloat16)


def apply_pkv_rerotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding, position_ids = None) -> DynamicCache:
    # this algorithm is to remove the RoPE on Ks (rotate them to start from position 0) 
    # input:   pkv: KV cache with PE
    #          position_ids: the original positions used to encode the PE
    device = pkv['key_cache'][0].device
    emb.to(device=device)
    if position_ids is None:  
        # default: start form 0
        position_ids = torch.arange(start=0, end=pkv['key_cache'][0].size(-2), dtype=torch.int64, device = device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv['key_cache'][0].size(0), 1]).to(device = device)
    cos, sin = emb(x=pkv['key_cache'][0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv['key_cache'])):
        new_device = pkv['key_cache'][i].device
        if new_device != device:
            emb.to(device=new_device)
            cos = cos.to(device = new_device)
            sin = sin.to(device = new_device)
            position_ids = position_ids.to(device=new_device)
            device = new_device
        pkv['key_cache'][i] = apply_rotary_pos_emb(
            k=pkv['key_cache'][i].to(dtype=pkv['value_cache'][0].dtype), cos=cos, sin=-sin, position_ids=position_ids
        )
    return pkv

def apply_pkv_rotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding, position_ids = None) -> DynamicCache:
    # this algorithm is to rotate RoPE to the desired positions, note the input pkv must be from position 0!!!
    # input:    pkv: KV cache with PE from 0
    #           position_ids: the target positions to rotate PE to
    device = pkv['key_cache'][0].device
    emb.to(device=device)
    if position_ids is None:
        position_ids = torch.arange(start=0, end=pkv['key_cache'][0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv['key_cache'][0].size(0), 1]).to(device = device)
    cos, sin = emb(x=pkv['key_cache'][0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv['key_cache'])):
        new_device = pkv['key_cache'][i].device
        if new_device != device:
            emb.to(device=new_device)
            cos = cos.to(device = new_device)
            sin = sin.to(device = new_device)
            position_ids = position_ids.to(device=new_device)
            device = new_device
        pkv['key_cache'][i] = apply_rotary_pos_emb(
            k=pkv['key_cache'][i].to(dtype=pkv['value_cache'][0].dtype), cos=cos, sin=sin, position_ids=position_ids
        )
    return pkv



# toll functions to cut KV

def cut_pkv(pkv, positions):
    # cut the KV cache, leave only elements on positions
    for layer_id in range(len(pkv['key_cache'])):
        pkv['key_cache'][layer_id] = pkv['key_cache'][layer_id][:,:,positions,:]
        pkv['value_cache'][layer_id] = pkv['value_cache'][layer_id][:,:,positions,:]
    return pkv

def divide_pkv(pkv, split_id):
    # divide the KV cache, return pkv before and after the split id
    after_pkv = type(pkv)()
    for layer_id in range(len(pkv['key_cache'])):
        after_pkv['key_cache'].append(pkv['key_cache'][layer_id][:, :, split_id:, :])
        after_pkv['value_cache'].append(pkv['value_cache'][layer_id][:, :, split_id:, :])
        pkv['key_cache'][layer_id] = pkv['key_cache'][layer_id][:, :, :split_id, :]
        pkv['value_cache'][layer_id] = pkv['value_cache'][layer_id][:, :, :split_id, :]
    return pkv, after_pkv
    
def flatten_pkv(pkv, mask):
    # flatten the pkv, remove the padding tokens
    for layer_id in range(len(pkv['key_cache'])):
        pkv['key_cache'][layer_id] = pkv['key_cache'][layer_id].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        pkv['value_cache'][layer_id] = pkv['value_cache'][layer_id].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
    return pkv

def stack_pkv(pkv, bsz):
    # stack the pkc towards batch_size
    for layer_id in range(len(pkv['key_cache'])):
        pkv['key_cache'][layer_id] = pkv['key_cache'][layer_id].repeat(bsz, 1, 1, 1)
        pkv['value_cache'][layer_id] = pkv['value_cache'][layer_id].repeat(bsz, 1, 1, 1)
    return pkv

def concact_pkv(pkv1, pkv2):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv2['key_cache'])):
        pkv2['key_cache'][layer_id] = torch.concat((pkv1['key_cache'][layer_id], pkv2['key_cache'][layer_id]), dim = 2)
        pkv2['value_cache'][layer_id] = torch.concat((pkv1['value_cache'][layer_id], pkv2['value_cache'][layer_id]), dim = 2)
    return pkv2

def concact_pkv_before(pkv1, pkv2):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv2['key_cache'])):
        pkv1['key_cache'][layer_id] = torch.concat((pkv1['key_cache'][layer_id].cpu(), pkv2['key_cache'][layer_id].cpu()), dim = 2)
        pkv1['value_cache'][layer_id] = torch.concat((pkv1['value_cache'][layer_id].cpu(), pkv2['value_cache'][layer_id].cpu()), dim = 2)
    return pkv1



def topk_pkv(pkv, top_k):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv['key_cache'])):
        pkv['key_cache'][layer_id] = pkv['key_cache'][layer_id][-top_k:, :, :, :]
        pkv['value_cache'][layer_id] = pkv['value_cache'][layer_id][-top_k:, :, :, :]
    return pkv

def init_empty_pkv(pkv_example, total_length):
    new_pkv = type(pkv_example)()
    new_pkv['key_cache'] = []
    new_pkv['value_cache'] = []
    for layer_id in range(len(pkv_example['key_cache'])):
        B, H, _, D = pkv_example['key_cache'][layer_id].shape
        new_pkv['key_cache'].append(torch.empty(B, H, total_length, D, dtype=pkv_example['key_cache'][layer_id].dtype, device=pkv_example['key_cache'][layer_id].device))
        new_pkv['value_cache'].append(torch.empty(B, H, total_length, D, dtype=pkv_example['value_cache'][layer_id].dtype, device=pkv_example['value_cache'][layer_id].device))
    return new_pkv

# def gapemp_graph(model, tokenizer, emb, prefix, center_node, neighbor_nodes):
#     # mode: inference or attention
#     with torch.no_grad():
#         prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
#         # query_input_ids = tokenizer(sentences, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
#         len_prefix = prefix_input_ids.shape[1]
#         # len_query = query_input_ids.shape[1]
#         center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
#         len_center = center_input_ids.shape[-1]
#         neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
#         len_neighbors = neighbors_input_ids.shape[-1]
#         len_flatten_neighbors = 0
#         neighbor_position_wo_prefix = []

#         for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
#             neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
#             len_flatten_neighbors+=neighbor_input_ids.shape[-1]
#             neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
#             neighbor_outputs = model(
#                 neighbor_input_ids.to(model.device),
#                 use_cache=True,)
#             tmp_pkv = neighbor_outputs.past_key_values
#             tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
#             expected_position = torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64)+len_prefix
#             tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
#             if neighbor_id==0:
#                 flatten_pkv = type(tmp_pkv)()
#                 flatten_pkv['key_cache'] = [t.clone().detach() for t in tmp_pkv['key_cache']]
#                 flatten_pkv['value_cache'] = [t.clone().detach() for t in tmp_pkv['value_cache']]
#             else:
#                 flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
#         neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
#         center_outputs = model(
#             center_input_ids.to(model.device),
#             past_key_values = flatten_pkv,
#             cache_position = torch.arange(start=0, end=len_center, dtype=torch.int64, device = model.device)+len_neighbors+len_prefix,
#             use_cache=True,
#         )

#         # flatten_pkv = center_outputs.past_key_values
        
#         # # prefix_outputs = model(
#         # #     prefix_input_ids.to(model.device),
#         # #     past_key_values=None,
#         # #     use_cache=True,
#         # # )
#         # # prefix_pkv = prefix_outputs.past_key_values
#         # # flatten_pkv = concact_pkv(prefix_pkv, flatten_pkv)
#         # generated = query_input_ids
#         # cache_position = torch.arange(start=0, end=len_query, dtype=torch.int64, device = model.device) + len_prefix + len_neighbors + len_center
#         # past_key_values = flatten_pkv
#         # num_generate = 0
#         # answer_ids = generated.to(model.device)
#         # while num_generate<256 and generated[0][0]!=tokenizer.eos_token_id:
#         #     outputs = model(generated.to(model.device), 
#         #                     past_key_values = past_key_values,
#         #                     cache_position = cache_position,
#         #                     use_cache = True)
#         return center_outputs

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    
def custom_nvembed_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print(position_ids)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # print(input_ids.shape)
        # print(seq_length)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            # print("inside")
            # print(position_ids)
            # print(seq_length)
            position_ids = position_ids.view(-1, seq_length).long()
            

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, inputs_embeds.dtype
            )
        else:
            # 4d mask is passed through the layers
            
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask
                # print("create 4d attention on my own")
            elif len(attention_mask.size()) == 2:
                # print("use the 2d attention then expand")
                attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def gen_mask(B, past_len, new_len, device):
    # B = 1
    # past_len = 10
    # new_len = 5
    # device = neighbor_input_ids['input_ids'].device  # wherever your tensors live

    # 1) Build a 2-D “global” mask of shape (B, past_len + new_len)
    #    (1’s everywhere if you want full attention to all tokens)
    global_mask = torch.ones(B, past_len + new_len, device=device, dtype=torch.bool)
    # print(global_mask)
    # 2) Turn it into a 4-D mask (B, 1, new_len, past_len+new_len):
    #    this lets each of the new_len queries attend to all past+new positions.
    mask_4d = _prepare_4d_attention_mask(
        global_mask,
        dtype=torch.float32,  # match your model’s dtype (e.g. float16)
        tgt_len=new_len
    )
    return mask_4d

def generate_rectangular_causal_mask(attn: torch.Tensor, prev_len: int) -> torch.Tensor:
    """
    attn:    [batch_size, num_heads, tgt_len, src_len]
    prev_len: number of 'previous' key positions at the LEFT of the src axis
    
    Returns a mask of shape [batch_size, num_heads, tgt_len, src_len] with:
      - mask[b,h,i,j] =   0.0   if j < prev_len
                          0.0   if j >= prev_len and (j - prev_len) <= i
                          -inf  otherwise
    """
    B, H, Lq, Lk = attn.shape
    device, dtype = attn.device, attn.dtype
    minf = torch.finfo(dtype).min

    # row indices (0…Lq-1) vs. column indices (0…Lk-1)
    rows = torch.arange(Lq, device=device).view(Lq, 1)       # [Lq, 1]
    cols = torch.arange(Lk, device=device).view(1, Lk)       # [1, Lk]

    # allowed if either
    #  1) in the "old" key region: j < prev_len
    #  2) in the "new" region but j_new_index = j - prev_len <= row index i
    allow = (cols < prev_len) | ((cols - prev_len) <= rows)  # [Lq, Lk]

    # build 2D mask then expand to 4D
    mask2d = torch.where(
        allow,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(minf, dtype=dtype, device=device)
    )                                                       # [Lq, Lk]

    # unsqueeze & expand to [B, H, Lq, Lk]
    return mask2d.view(1, 1, Lq, Lk).expand(B, H, Lq, Lk)
    
def gapemp_graph_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, N, D, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=True, return_tensors="pt", add_special_tokens=False, max_length=5000).input_ids
        # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        len_prefix = prefix_input_ids.shape[1]
        # len_query = query_input_ids.shape[1]
        middle = ''
        embed_list = []
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            print(idx)
            center_node = middle + center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, add_special_tokens=False, max_length=5000).to(model.device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != []:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, add_special_tokens=False, max_length=5000).to(model.device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]
            else:
                len_neighbors=0
            

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, add_special_tokens=False, max_length=5000).to(model.device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model.embedding_model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)+len_prefix
                    tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)

                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i], value_cache[i])
                    for i in range(len(key_cache))
                )
            else:
                flatten_pkv_tuple = None

            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = model.device)+len_neighbors+len_prefix
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())

            past_key_values = flatten_pkv_tuple
            seq_length = center_input_ids['input_ids'].shape[-1]
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            # print(pos)
            # model.embedding_model.forward = types.MethodType(custom_nvembed_forward, model.embedding_model)
            center_outputs = model.embedding_model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            
            print(idx)

            # embeds = model.latent_attention_model(
            #     center_outputs.last_hidden_state,
            #     center_input_ids['attention_mask']
            # )
            # print(center_outputs.last_hidden_state.size())
            embeds = center_outputs.last_hidden_state.mean(dim=(0,1))
            embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            del embeds
            torch.cuda.empty_cache()
            idx += 1
            
            # flatten_pkv = center_outputs.past_key_values
            # if cursor == 0:
            #     overall_pkv = type(flatten_pkv)()
            #     overall_pkv['key_cache'] = [t.clone().detach() for t in flatten_pkv['key_cache']]
            #     overall_pkv['value_cache'] = [t.clone().detach() for t in flatten_pkv['value_cache']]
            # else:
            #     overall_pkv = concact_pkv_before(overall_pkv, flatten_pkv)
            # cursor = max(cursor, len_neighbors + len_center)
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        return torch.cat(embed_list, dim=0)
    
def find_cap(lengths, total_limit=60000, element_limit=8192):
    """
    lengths: list of original lengths (ints)
    total_limit: maximum allowed sum
    element_limit: hard cap for any single element
    returns: cap T
    """
    # 1. First clip to element_limit, then sort
    clipped = [min(l, element_limit) for l in lengths]
    sorted_l = sorted(clipped)
    n = len(sorted_l)

    # 2. Build prefix sums: prefix[i] = sum(sorted_l[0:i])
    prefix = [0]
    for l in sorted_l:
        prefix.append(prefix[-1] + l)

    # 3. Try each “breakpoint” k, meaning k items are below T,
    #    the remaining (n-k) items are all set to T.
    for k in range(n + 1):
        used = prefix[k]                # sum of the k smallest items
        rem = total_limit - used        # budget left for the other (n-k) items
        cnt = n - k
        # if no items left, any T ≥ sorted_l[-1] works
        if cnt == 0:
            # if we haven’t already exceeded, we can take the max cap
            if used <= total_limit:
                return element_limit
            else:
                continue

        # candidate cap
        T = rem / cnt

        # the “valid interval” for T is [sorted_l[k-1], sorted_l[k]]
        lo = sorted_l[k - 1] if k > 0 else 0
        hi = sorted_l[k]

        if lo <= T <= hi and T <= element_limit:
            return T

    # Fallback: if even the average is above element_limit, just use element_limit
    return element_limit

# def free_non_model_tensors(model):
#     # 1) Gather the IDs of every tensor you want to keep:
#     keep = set(id(t) for t in model.parameters())
#     keep |= set(id(b) for b in model.buffers())

#     # 2) Find all live CUDA tensors:
#     live = [o for o in gc.get_objects()
#             if isinstance(o, torch.Tensor) and o.device.type == "cuda"]

#     # 3) Separate out the ones you want to free:
#     to_free = [t for t in live if id(t) not in keep]
#     # print(f"Freeing {len(to_free)} non‐model CUDA tensors")

#     # 4a) If you truly want them gone, break *all* references:
#     #    This can get tricky if they live inside unnamed closures or other data structures.
#     #    A simple trick is to overwrite their storage by moving them to CPU:
#     for t in to_free:
#         # detach so you don’t keep a grad‐graph ref
#         cpu_copy = t.detach().cpu()
#         # reassign the tensor’s data pointer to the small CPU copy
#         t.data = cpu_copy.data
#         # drop your handle to the CPU copy
#         del cpu_copy

#     # 4b) Now force Python to GC and clear PyTorch’s cache:
#     gc.collect()
#     torch.cuda.empty_cache()

    # 5) Optionally, check PyTorch’s view of “live” allocations:
    # print("  PyTorch reports allocated:",
    #       torch.cuda.memory_allocated() / 1024**2, "MiB")

def free_non_model_tensors(model):
    # 1) IDs of tensors you want to keep (model params + buffers)
    keep = {id(p) for p in model.parameters()}
    keep |= {id(b) for b in model.buffers()}

    live_cuda_tensors = []
    for o in gc.get_objects():
        try:
            # Some gc objects are weakref proxies; even isinstance() can raise
            if isinstance(o, torch.Tensor):
                t = o
            else:
                # Skip anything that's not a Tensor (avoid touching .data if possible)
                continue

            # Now safe to check device (guard anyway)
            if t.is_cuda:
                live_cuda_tensors.append(t)
        except ReferenceError:
            # Referent vanished; ignore
            continue
        except Exception:
            # Be defensive against any odd object types
            continue

    # 2) Free the CUDA tensors that aren't model-owned
    to_free = [t for t in live_cuda_tensors if id(t) not in keep]

    # Best you can do is drop references; if something else still holds them,
    # GC won't reclaim. Also detach to sever graph references.
    # for t in to_free:
    #     try:
    #         if t.grad_fn is not None or t.requires_grad:
    #             t = t.detach()
    #         # Drop any views to encourage freeing storage
    #         del t
    #     except Exception:
    #         pass
    
    for t in to_free:
        # detach so you don’t keep a grad‐graph ref
        cpu_copy = t.detach().cpu()
        # reassign the tensor’s data pointer to the small CPU copy
        t.data = cpu_copy.data
        # drop your handle to the CPU copy
        del cpu_copy

    # 3) Collect & release CUDA caching allocator
    gc.collect()
    torch.cuda.empty_cache()

import subprocess
def print_full_gpu_usage(tag="", i=4):
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(i),
         "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        encoding="utf-8"
    )
    used, total = [x.strip() for x in out.strip().split(",")]
    print(f"[GPU {i}][{tag}] {used} MiB / {total} MiB")
    
def gapemp_graph_batch_qwen(tokenizer, model, emb, scale, cap, center_node_list, neighbor_nodes_list, N, D, max_token, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), cap, element_limit=max_token))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] * scale  for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] * scale for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            seq_length = center_input_ids['input_ids'].shape[-1]
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(seq_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(pos)
            # model.embedding_model.forward = types.MethodType(custom_nvembed_forward, model.embedding_model)
            # print("ready to encode center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("input center")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print('ffff')
            
            # embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            # print(center_outputs.last_hidden_state.size())
            # embeds = center_outputs.last_hidden_state.mean(dim=(0,1))

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)

def gapemp_graph_batch_qwen_time(tokenizer, model, emb, scale, cap, center_node_list, neighbor_nodes_list, N, D, max_token, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        times = []
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), cap, element_limit=max_token))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] * scale  for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] * scale for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            seq_length = center_input_ids['input_ids'].shape[-1]
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(seq_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(pos)
            # model.embedding_model.forward = types.MethodType(custom_nvembed_forward, model.embedding_model)
            # print("ready to encode center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("input center")
            torch.cuda.synchronize()
            start = time.time()
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print('ffff')
            
            # embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            # print(center_outputs.last_hidden_state.size())
            # embeds = center_outputs.last_hidden_state.mean(dim=(0,1))

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            torch.cuda.synchronize()
            end = time.time()
            if idx >= idx_start+5:
                times.append(end - start)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)
        if times:
            avg = sum(times) / len(times)
            print(f"\nStats over {len(times)} passages (after 5 warm-up):")
            print(f"Average: {avg:.4f} sec, Min: {min(times):.4f}, Max: {max(times):.4f}")


def gapemp_graph_qwen_context_agg(tokenizer, model, emb, context_instruct, cap, scale, center_node_list, neighbor_nodes_list, N, D, max_token, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), cap, element_limit=max_token))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            
            # start to get the context key and values
            context_input_ids = tokenizer(context_instruct, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_context = context_input_ids['input_ids'].shape[-1]

            pos = torch.arange(start=0, end=len_context, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(context_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(len_context)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(context_input_ids['input_ids'].shape[0], past_key_values_length, len_context, context_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)

            # print_full_gpu_usage("context embedding")
            context_outputs = model(
                input_ids = context_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print_full_gpu_usage("context embedding finish")
            # cpu_copy = tuple(
            #     tuple(kv.detach().cpu() for kv in layer)
            #     for layer in flatten_pkv_tuple
            # )
            # flatten_pkv_tuple.data = cpu_copy.data
            # del cpu_copy
            # print_full_gpu_usage("cpu previous kv")
            # del flatten_pkv_tuple
            # print_full_gpu_usage("cpu previous kv")

            tmp_pkv = {}
            tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
            free_non_model_tensors(model)
            # print_full_gpu_usage("clear everything")
            key_cache   = tmp_pkv["key_cache"]
            value_cache = tmp_pkv["value_cache"]
            tmp_pkv = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
            # print_full_gpu_usage("save context kv")
            
            # encode the center node
            center_input_ids.to(device)
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_context
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(tmp_pkv, Cache)
            if use_legacy_cache:
                tmp_pkv = DynamicCache.from_legacy_cache(tmp_pkv)
            past_key_values_length = tmp_pkv.get_usable_length(len_center)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(past_key_values_length)
            # print(overall_mask.shape)

            # print_full_gpu_usage("center embedding")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = tmp_pkv,
                position_ids = pos,
                use_cache=True,
            )

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)

def gapemp_graph_qwen_context_agg_fixed(tokenizer, model, emb, context_instruct, cap, scale, center_node_list, neighbor_nodes_list, N, D, max_token, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), cap, element_limit=max_token))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                # del neighbor_input_ids, neighbor_outputs, flatten_pkv         
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            
            # start to get the context key and values
            context_input_ids = tokenizer(context_instruct, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_context = context_input_ids['input_ids'].shape[-1]

            pos = torch.arange(start=0, end=len_context, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(context_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(len_context)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(context_input_ids['input_ids'].shape[0], past_key_values_length, len_context, context_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(past_key_values_length)
            # print_full_gpu_usage("context embedding")
            context_outputs = model(
                input_ids = context_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print_full_gpu_usage("context embedding finish")
            # cpu_copy = tuple(
            #     tuple(kv.detach().cpu() for kv in layer)
            #     for layer in flatten_pkv_tuple
            # )
            # flatten_pkv_tuple.data = cpu_copy.data
            # del cpu_copy
            # print_full_gpu_usage("cpu previous kv")
            # del flatten_pkv_tuple
            # print_full_gpu_usage("cpu previous kv")

            tmp_pkv = {}
            tmp_pkv['key_cache'] = [layer_kv[0][:, :, -len_context:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv['value_cache'] = [layer_kv[1][:, :, -len_context:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
            tmp_pkv = concact_pkv_before(tmp_pkv, flatten_pkv)
            free_non_model_tensors(model)
            # print_full_gpu_usage("clear everything")

            key_cache   = tmp_pkv["key_cache"]
            value_cache = tmp_pkv["value_cache"]
            tmp_pkv = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
            # print_full_gpu_usage("save context kv")
            
            # encode the center node
            center_input_ids.to(device)
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ max(len_neighbors, len_context-2)
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(len_neighbors)
            # print(pos)
            use_legacy_cache = not isinstance(tmp_pkv, Cache)
            if use_legacy_cache:
                tmp_pkv = DynamicCache.from_legacy_cache(tmp_pkv)
            past_key_values_length = tmp_pkv.get_usable_length(len_center)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(len_context)
            # print(past_key_values_length)
            # print(len_all_neighbors)
            # print(past_key_values_length)
            # print(past_key_values_length)
            # print(overall_mask.shape)

            # print_full_gpu_usage("center embedding")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = tmp_pkv,
                position_ids = pos,
                use_cache=True,
            )

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)
    
def gapemp_graph_qwen_context_agg_new(tokenizer, model, emb, context_instruct, cap, scale, center_node_list, neighbor_nodes_list, N, D, max_token, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), cap, element_limit=max_token))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            # print(flatten_pkv_tuple[0][0].shape)
            
            # start to get the context key and values
            context_input_ids = tokenizer(context_instruct, return_tensors='pt', truncation=True, padding=True, max_length = max_token).to(device)
            len_context = context_input_ids['input_ids'].shape[-1]

            pos = torch.arange(start=0, end=len_context, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(context_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(len_context)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(context_input_ids['input_ids'].shape[0], past_key_values_length, len_context, context_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(overall_mask.shape)
            # print(overall_mask.shape)
            # print(past_key_values_length)

            # print_full_gpu_usage("context embedding")
            context_outputs = model(
                input_ids = context_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print_full_gpu_usage("context embedding finish")
            # cpu_copy = tuple(
            #     tuple(kv.detach().cpu() for kv in layer)
            #     for layer in flatten_pkv_tuple
            # )
            # flatten_pkv_tuple.data = cpu_copy.data
            # del cpu_copy
            # print_full_gpu_usage("cpu previous kv")
            # del flatten_pkv_tuple
            # print_full_gpu_usage("cpu previous kv")

            tmp_pkv = {}
            # print(context_outputs.past_key_values[0][0][:, :, 0:-2, :].shape)
            # print(context_outputs.past_key_values[0][0].shape)
            tmp_pkv['key_cache'] = [layer_kv[0][:, :, -len_context:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv['value_cache'] = [layer_kv[1][:, :, -len_context:-2, :].cpu() * scale for layer_kv in context_outputs.past_key_values]
            tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
            free_non_model_tensors(model)
            # print_full_gpu_usage("clear everything")
            key_cache   = tmp_pkv["key_cache"]
            value_cache = tmp_pkv["value_cache"]
            tmp_pkv = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
            # print_full_gpu_usage("save context kv")
            
            # encode the center node
            center_input_ids.to(device)
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_context -2
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(tmp_pkv, Cache)
            if use_legacy_cache:
                tmp_pkv = DynamicCache.from_legacy_cache(tmp_pkv)
            past_key_values_length = tmp_pkv.get_usable_length(len_center)
            # print(past_key_values_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))
            # print(past_key_values_length)
            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(overall_mask.shape)
            # print(overall_mask.shape)
            # print(tmp_pkv[0][0].shape)

            # print_full_gpu_usage("center embedding")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = tmp_pkv,
                position_ids = pos,
                use_cache=True,
            )

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)

def gapemp_graph_qwen_condense(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, N, D, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    # max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), 20000))
                    max_length = 8192
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            seq_length = center_input_ids['input_ids'].shape[-1]
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(seq_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(pos)
            # model.embedding_model.forward = types.MethodType(custom_nvembed_forward, model.embedding_model)
            # print("ready to encode center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("input center")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print('ffff')
            
            # embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            # print(center_outputs.last_hidden_state.size())
            # embeds = center_outputs.last_hidden_state.mean(dim=(0,1))

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)

def gapemp_graph_qwen_neighbor_embedding(tokenizer, model, emb, context_instruct, neighbor_nodes_list, N, D, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for neighbor_nodes in neighbor_nodes_list:
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            # center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
            # len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), 55000))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            
            # start to get the context key and values
            context_input_ids = tokenizer(context_instruct, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
            len_context = context_input_ids['input_ids'].shape[-1]

            pos = torch.arange(start=0, end=len_context, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(context_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(len_context)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(context_input_ids['input_ids'].shape[0], past_key_values_length, len_context, context_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)

            # print_full_gpu_usage("context embedding")
            context_outputs = model(
                input_ids = context_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )

            embeds = last_token_pool(context_outputs.last_hidden_state, context_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            # del embeds
            # if 'neighbors_input_ids' in locals():
            #     del neighbors_input_ids
            # del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)

def gapemp_graph_qwen_ape(tokenizer, model, emb, temperature, scale, center_node_list, neighbor_nodes_list, N, D, device, filename, idx_start = 0):
    # mode: inference or attention
    with torch.no_grad():
        # prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        # # query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        # len_prefix = prefix_input_ids.shape[1]
        # # len_query = query_input_ids.shape[1]
        # middle = ''
        idx = idx_start

        # N = 30000          # total size
        # D = 4096  # if wrapped in DataParallel
        emb_file = np.memmap(
            f"{filename}_embedding_{idx_start}.npy",
            dtype="float32",
            mode="w+",
            shape=(N, D)
        )
        # print("Before:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            # print_full_gpu_usage("before anything")
            # center_node =  center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
            len_center = center_input_ids['input_ids'].shape[-1]
            if neighbor_nodes != '' and len(neighbor_nodes) > 0:
                neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
                len_neighbors = neighbors_input_ids['input_ids'].shape[-1]-2
                # del neighbors_input_ids
            else:
                len_neighbors=0

            if len_neighbors > 0:
                for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                    # max_length = 1000 if len(neighbor_nodes) > 10 else 5000
                    # max_length = int(2000 / (int((len(neighbor_nodes)-1)/10)+1))
                    # if len(neighbor_nodes) > 40:
                    #     max_length = 200
                    # max_length = 1000
                    # if len_neighbors * len(neighbor_nodes) > 81920:
                    #     max_length = int(80000/len(neighbor_nodes))
                    # else:
                    #     max_length = 8192
                    
                    max_length = int(find_cap(neighbors_input_ids['attention_mask'].sum(dim=1), 55000))
                    neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=True, padding=True, max_length = max_length).to(device)
                    # len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                    # neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                    neighbor_outputs = model(
                        input_ids = neighbor_input_ids["input_ids"],
                        attention_mask = neighbor_input_ids["attention_mask"],
                        use_cache=True,)
                    # outputs = model(**batch_dict)
                    tmp_pkv = {}
                    tmp_pkv['key_cache'] = [layer_kv[0][:, :, 0:-2, :] / temperature  for layer_kv in neighbor_outputs.past_key_values]
                    tmp_pkv['value_cache'] = [layer_kv[1][:, :, 0:-2, :] * scale for layer_kv in neighbor_outputs.past_key_values]
                    # tmp_pkv = neighbor_outputs.past_key_values

                    # tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                    # expected_position = torch.arange(start=0, end=neighbor_input_ids['input_ids'].shape[-1], dtype=torch.int64)
                    # tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                    # print('g')
                    if neighbor_id==0:
                        flatten_pkv = type(tmp_pkv)()
                        flatten_pkv['key_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['key_cache']]
                        flatten_pkv['value_cache'] = [t.clone().detach().cpu() for t in tmp_pkv['value_cache']]
                    else:
                        flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
                    # print("iteration:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                    # print_full_gpu_usage("iteration")
                # neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
                # print('ffff')
                key_cache   = flatten_pkv["key_cache"]
                value_cache = flatten_pkv["value_cache"]
                # rebuild past_key_values
                flatten_pkv_tuple = tuple(
                    (key_cache[i].to(device), value_cache[i].to(device))
                    for i in range(len(key_cache))
                )
                # print("place 1:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
                # print_full_gpu_usage("loop neighbor over")
                del neighbor_input_ids, neighbor_outputs, flatten_pkv
                
            else:
                flatten_pkv_tuple = None

            # print("start center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("start center")
            # print('ffff')
            pos = torch.arange(start=0, end=len_center, dtype=torch.int64, device = device)+ len_neighbors
            pos = pos.unsqueeze(0).expand(center_input_ids['input_ids'].shape[0], -1)
            # print(pos.view(-1, center_input_ids['input_ids'].shape[-1]).long())
            # print(pos)
            seq_length = center_input_ids['input_ids'].shape[-1]
            use_legacy_cache = not isinstance(flatten_pkv_tuple, Cache)
            if use_legacy_cache:
                flatten_pkv_tuple = DynamicCache.from_legacy_cache(flatten_pkv_tuple)
            past_key_values_length = flatten_pkv_tuple.get_usable_length(seq_length)
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.device
            # )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # print(position_ids)
            # print("center length: " + str(len_center) + "  ; neighbor length: " + str(len_neighbors))

            overall_mask = gen_mask(center_input_ids['input_ids'].shape[0], past_key_values_length, len_center, center_input_ids['input_ids'].device)
            overall_mask = generate_rectangular_causal_mask(overall_mask, prev_len=past_key_values_length)
            # print(pos)
            # model.embedding_model.forward = types.MethodType(custom_nvembed_forward, model.embedding_model)
            # print("ready to encode center:", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("input center")
            center_outputs = model(
                input_ids = center_input_ids['input_ids'],
                attention_mask = overall_mask,
                past_key_values = flatten_pkv_tuple,
                position_ids = pos,
                use_cache=True,
            )
            # print('ffff')
            
            # embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            # print(center_outputs.last_hidden_state.size())
            # embeds = center_outputs.last_hidden_state.mean(dim=(0,1))

            embeds = last_token_pool(center_outputs.last_hidden_state, center_input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            # embed_list.append(embeds)
            emb_file[idx] = embeds.cpu().numpy()
            print(idx)
            # print("finish embedding", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("finish embedding")

            del embeds
            del center_input_ids
            if 'neighbors_input_ids' in locals():
                del neighbors_input_ids
            del pos, overall_mask, center_outputs
            # print("clean delete", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("delete")
            gc.collect()
            # print("clean gc", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("gc")
            torch.cuda.empty_cache()
            # print("clean cache", torch.cuda.memory_reserved() / 1024**2, "MB allocated")
            # print_full_gpu_usage("cache")

            free_non_model_tensors(model)
            # print_full_gpu_usage("live clear")

            # live = [obj for obj in gc.get_objects()
            # if isinstance(obj, torch.Tensor) and obj.device.type=="cuda"]
            # print("Still-alive CUDA tensors:", live)
            idx += 1
            
        emb_file.flush()
        print(f"Wrote {idx}×{D} embeddings to embeddings.npy")
        # return torch.cat(embed_list, dim=0)