import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
import random
import time
from tabulate import tabulate
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
import os
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)
    
def global_merge_2d(
        metric: torch.Tensor,
        w: int, h: int, sx: int, sy: int, 
        reduce_num: int,
        no_rand: bool = False,
        generator: torch.Generator = None
        ) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, C = metric.shape

    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=metric.device)
            # rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        reduce_num = min(a.shape[1], reduce_num)
        reduce_num = reduce_num // 16 * 16

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., reduce_num:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :reduce_num, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        dst_in_x_index = b_idx.expand(B, -1, C)
        unm_in_x_index = gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, -1, C)
        src_in_x_index = gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, -1, C)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, dst = split(x)
        n, t1, c = a.shape
        
        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))
        src = gather(a, dim=-2, index=src_idx.expand(n, reduce_num, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, reduce_num, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)
    

    def mprune(x: torch.Tensor) -> torch.Tensor:
        a, dst = split(x)
        n, t1, c = a.shape
        
        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))

        return torch.cat([unm, dst], dim=1)
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # NOTE: a_idx is (a in x) b_idx is (dst in x), 
        # NOTE: dst_idx is (src in dst), unm_idx is (unm in a), (src_idx) is (src in a)

        out.scatter_(dim=-2, index=dst_in_x_index, src=dst)
        out.scatter_(dim=-2, index=unm_in_x_index, src=unm)
        out.scatter_(dim=-2, index=src_in_x_index, src=src)
        return out
    
    return merge, mprune, unmerge


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tore_info"):
            for hook in module._tore_info["hooks"]:
                hook.remove()
            module._tore_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model

def compute_merge(x: torch.Tensor, tore_info: Dict[str, Any]) -> Tuple[Callable, ...]:

    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"
    ratio_current = tore_info["states"]["ratio_current"]

    reduce_num = int(x.shape[1] * (1 - ratio_current))

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if tore_info["args"]["generator"] is None:
        tore_info["args"]["generator"] = init_generator(x.device)
    elif tore_info["args"]["generator"].device != x.device:
        tore_info["args"]["generator"] = init_generator(x.device, fallback=tore_info["args"]["generator"])

    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else tore_info["args"]["use_rand"]

    m, mp, u  = global_merge_2d(x, w, h, tore_info["args"]["sx"], tore_info["args"]["sy"], reduce_num=reduce_num, 
                                no_rand=not use_rand, generator=tore_info["args"]["generator"])

    if tore_info["args"]["change_merge_to_prune"]:
        m_a, u_a = (mp, u) if tore_info["args"]["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (mp, u) if tore_info["args"]["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (mp, u) if tore_info["args"]["merge_mlp"]       else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if tore_info["args"]["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (m, u) if tore_info["args"]["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if tore_info["args"]["merge_mlp"]       else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good



def make_ToMe_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class StableDiffusion3Pipeline_ToMe(pipe_class):
        # Save for unpatching later
        _parent = pipe_class

        def __call__(self, *args, **kwargs):
            self._tore_info["states"]["step_count"] = kwargs['num_inference_steps']
            self._tore_info["states"]["step_iter"] = list(range(kwargs['num_inference_steps']))
            output = super().__call__(*args, **kwargs)
            return output

    return StableDiffusion3Pipeline_ToMe

def make_ToMe_model(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    
    class SD3Transformer2DModel_ToMe(model_class):
        _parent = model_class

        def forward(self, *args, **kwargs):
            self._tore_info["states"]["layer_count"] = self.config.num_layers
            self._tore_info["states"]["step_current"] = self._tore_info["states"]["step_iter"].pop(0)
            self._tore_info["states"]["layer_iter"] = list(range(self.config.num_layers))
            output = super().forward(*args, **kwargs)
            return output

    return SD3Transformer2DModel_ToMe

def make_ToMe_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class JointTransformerBlock_ToMe(block_class):
        
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):

            self._tore_info["states"]["layer_current"] = self._tore_info["states"]["layer_iter"].pop(0)
            step_current = self._tore_info["states"]["step_current"]
            layer_current = self._tore_info["states"]["layer_current"]

            joint_attention_kwargs = joint_attention_kwargs or {}
            if self.use_dual_attention:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                    hidden_states, emb=temb
                )
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
            #! Step 1: Compute_Merge
            m_a, _, m_m, u_a, _, u_m = compute_merge(norm_hidden_states, self._tore_info)
            if self.use_dual_attention:
                m_a2, _, _, u_a2, _, _ = compute_merge(norm_hidden_states2, self._tore_info)

            #! Step 2_1: Merge_Attn
            norm_hidden_states = m_a(norm_hidden_states)

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                **joint_attention_kwargs,
            )
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output

            #! Step 2_2: UnMerge_Attn
            attn_output = u_a(attn_output)

            hidden_states = hidden_states + attn_output

            if self.use_dual_attention:
                #! Step 2_3: Merge_DualAttn
                norm_hidden_states2 = m_a2(norm_hidden_states2)

                attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
                attn_output2 = gate_msa2.unsqueeze(1) * attn_output2

                #! Step 2_4: UnMerge_DualAttn
                attn_output2 = u_a2(attn_output2)

                hidden_states = hidden_states + attn_output2

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            #! Step 3_1: Merge_MLP
            norm_hidden_states = m_m(norm_hidden_states)

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            #! Step 3_2: UnMerge_MLP
            ff_output = u_m(ff_output)

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
                
            return encoder_hidden_states, hidden_states
    
    return JointTransformerBlock_ToMe

def apply_ToMe(
        pipe: torch.nn.Module,
        ratio: float = 0.5,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        change_merge_to_prune: bool = False,
        ):

    # Make sure the module is not currently patched
    remove_patch(pipe)
    make_pipe_fn = make_ToMe_pipe
    pipe.__class__ = make_pipe_fn(pipe.__class__)
    
    pipe._tore_info = {
        "type": "ToMe",
        "args":{
            "ratio": ratio,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "change_merge_to_prune": change_merge_to_prune,
            "generator": None,
        },
        "features":{
        },
        "states":{
            "ratio_current": ratio,
            "step_count": None,
            "step_iter": None,
            "step_current": None,
            "layer_count": None,
            "layer_iter": None,
            "layer_current": None,
        }
    }
    
    model = pipe.transformer
    make_model_fn = make_ToMe_model
    model.__class__ = make_model_fn(model.__class__)
    model._tore_info = pipe._tore_info
    for _, module in model.named_modules():
        if isinstance_str(module, "JointTransformerBlock"):
            make_block_fn = make_ToMe_block
            module.__class__ = make_block_fn(module.__class__)
            module._tore_info = pipe._tore_info
    return pipe
