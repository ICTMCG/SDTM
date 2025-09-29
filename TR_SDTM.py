import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward, Attention
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

def SSM(
    metric: torch.Tensor,
    reduce_num: int = 0,
    threshold: float = 0,
    window_size: Tuple[int, int] = (2,2),
    no_rand: bool = False,
    generator: torch.Generator = None,
    tore_info: Dict = None
) -> Tuple[Callable, Callable]:
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w
        assert num_token_window > 1, "window_size must produce at least 2 tokens (K>1)."
        
        B, N, D = metric.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        # metric = rearrange(metric, "b (h w) c -> b c h w", h=base_grid_H)
        metric = metric.view(-1, base_grid_H, base_grid_W, D).permute(0, 3, 1, 2)
    
        # metric = rearrange(metric, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//ws_h, gw=base_grid_W//ws_w)
        metric = metric.view(B, D, base_grid_H // ws_h, ws_h, base_grid_W // ws_w, ws_w).permute(0, 2, 4, 1, 3, 5)
        b, gh, gw, c, ps_h, ps_w = metric.shape

        # Flatten mxm window for pairwise operations
        tensor_flattened = metric.reshape(b, gh, gw, c, -1)
    
        # Expand dims for pairwise operations
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)

        # Compute cosine similarities
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        # Average similarities (excluding the self-similarity)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w))
            
        # similarity_map = rearrange(similarity_map.unsqueeze(1), 'b c h w-> b (c h w)')
        similarity_map = similarity_map.unsqueeze(1).reshape(similarity_map.size(0), -1)

        # ---- Frequency priority score integration ----
        ssmscore_map = similarity_map
        if tore_info is not None and "states" in tore_info and tore_info["states"].get("last_independent") is not None:
            li = tore_info["states"]["last_independent"]  # [B, N]
            if li.shape[1] == base_grid_H * base_grid_W:
                eps = 1e-6
                li_f = li.to(similarity_map.dtype)
                mean_li = li_f.mean(dim=1, keepdim=True) + eps
                indiv_priority = li_f / mean_li  # [B, N]
                indiv_priority_grid = indiv_priority.view(B, base_grid_H, base_grid_W)
                indiv_priority_windows = (
                    indiv_priority_grid
                    .view(B, gh, ws_h, gw, ws_w)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(B, gh, gw, ws_h * ws_w)
                    .mean(-1)
                )  # [B, gh, gw]
                indiv_priority_flat = indiv_priority_windows.view(B, gh * gw)
                a_s = tore_info.get("args", {}).get("a_s", 0.0)
                ssmscore_map = similarity_map + a_s * indiv_priority_flat
        # ----------------------------------------------
        
        #--- adaptive section ---#
        if reduce_num is None:
            n_B, n_H = ssmscore_map.shape
            node_mean = torch.tensor(threshold).cuda(sims.device)
            node_mean=node_mean.repeat(1,n_H)
            reduce_num = torch.ge(ssmscore_map, node_mean).sum(dim=1).min()
        else:
            reduce_num = reduce_num // 48 * 16 

        # -------------# 
    
        #   get top k similar super patches 
        _, sim_super_patch_idxs = ssmscore_map.topk(reduce_num, dim=-1)
    
        # --- creating the mergabel and unmergable super  pathes
        tensor = torch.arange(base_grid_H * base_grid_W, device=metric.device).reshape(base_grid_H, base_grid_W)

        # Repeat the tensor to create a batch of size 2
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        
        # Apply unfold operation on last two dimensions to create the sliding window
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)

        # Reshape the tensor to the desired shape 
        windowed_tensor = windowed_tensor.reshape(B, -1, num_token_window)
    
        # Use torch.gather to collect the desired elements
        gathered_tensor = torch.gather(windowed_tensor, 1, sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, num_token_window))

        # Create a mask for all indices, for each batch
        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool, device=metric.device)

        # Create a tensor that matches the shape of indices and fill it with False
        mask_values = torch.zeros_like(sim_super_patch_idxs, dtype=torch.bool, device=metric.device)

        # Use scatter_ to update the mask. This will set mask[b, indices[b]] = False for all b
        mask.scatter_(1, sim_super_patch_idxs, mask_values)

        # Get the remaining tensor
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, num_token_window)].reshape(B, -1, num_token_window)
        # unm_idx = remaining_tensor.reshape(B, -1).sort(dim=-1).values.unsqueeze(-1)
        unm_idx = remaining_tensor.reshape(B, -1).unsqueeze(-1)

        # --- Randomly choose dst inside each selected window (optimized version) --- #
        # Avoid boolean masking + reshape; use a precomputed src index table per possible dst.
        K = num_token_window
        dim_index = K - 1  # number of src tokens per window

        if no_rand:
            rand_pos = torch.zeros(B, reduce_num, 1, dtype=torch.long, device=metric.device)
        else:
            if generator is not None:
                rand_pos = torch.randint(K, (B, reduce_num, 1), device=metric.device, generator=generator)
            else:
                rand_pos = torch.randint(K, (B, reduce_num, 1), device=metric.device)

        # Precompute src position table (vectorized) under assumption K>1
        full = torch.arange(K, device=metric.device)
        matrix = full.unsqueeze(0).expand(K, K)    # [K,K]
        mask = torch.eye(K, dtype=torch.bool, device=metric.device)
        src_table = matrix[~mask].view(K, K-1)     # [K, K-1]

        # gathered_tensor: [B, reduce_num, K]; select dst
        dst_idx = torch.gather(gathered_tensor, 2, rand_pos).reshape(B, -1).unsqueeze(-1)

        # src positions (per window)
        src_pos = src_table[rand_pos.squeeze(-1)]            # [B, reduce_num, K-1]
        src_vals = torch.gather(gathered_tensor, 2, src_pos) # [B, reduce_num, K-1]
        src_idx = src_vals.reshape(B, reduce_num * dim_index).unsqueeze(-1)
        merge_idx = torch.arange(reduce_num, device=metric.device).repeat_interleave(dim_index).repeat(B, 1).unsqueeze(-1)

        independent_idx = None
        unindependent_idx = None

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
            unindependent_idx = src_idx.squeeze(-1)
        else:
            independent_idx = unm_idx.squeeze(-1)
            unindependent_idx = torch.cat([src_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
        update_last_independent(independent_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # TODO: num_token_window can be undefined
        
            n, t1, c = x.shape
            # src = x.gather(dim=-2, index=src_idx.expand(n, r*dim_index, c))
            # dst = x.gather(dim=-2, index=dst_idx.expand(n, r, c))
            # unm = x.gather(dim=-2, index=unm_idx.expand(n, t1 - (r*num_token_window), c))
            src = gather(x, dim=-2, index=src_idx.expand(n, reduce_num*dim_index, c))
            dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num*num_token_window), c))
            dst = dst.scatter_reduce(-2, merge_idx.expand(n,reduce_num*dim_index, c), src, reduce=mode)
            x = torch.cat([unm, dst], dim=1)

            return x
        
        def mprune(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # TODO: num_token_window can be undefined
            n, t1, c = x.shape

            dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num*num_token_window), c))
            x = torch.cat([unm, dst], dim=1)

            return x

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            # Determine cache_name directly from phase state; align with features dict keys
           

            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, tu, c = unm.shape
            # Reconstruct src from dst buckets (value copy, not true inverse)
            src = gather(dst, dim=-2, index=merge_idx.expand(B, reduce_num*dim_index, c))

            # Optional fusion with cached features for dst/src tokens
            # unm: keep as-is; dst/src: fuse computed with cached by mcw
            try:
                cache_name = tore_info["states"].get("unmerge_phase", None)
                if cache_name not in ("attn_output", "attn_output2", "mlp_output"):
                    pass
                mcw = float(tore_info.get("args", {}).get("mcw", 1.0))
                l = tore_info.get("states", {}).get("layer_current", -1)
                key = f"l{l}"
                cache_dict = tore_info.get("features", {}).get(cache_name, None)
                cache_full = cache_dict.get(key) if isinstance(cache_dict, dict) else None
                if cache_full is not None:
                    cache_full = cache_full.to(device=x.device, dtype=x.dtype)
                    # gather cached dst/src by original indices
                    cached_dst = gather(cache_full, dim=1, index=dst_idx.expand(B, reduce_num, c))
                    cached_src = gather(cache_full, dim=1, index=src_idx.expand(B, reduce_num*dim_index, c))
                    # fuse
                    dst = mcw * dst + (1.0 - mcw) * cached_dst
                    src = mcw * src + (1.0 - mcw) * cached_src
            except Exception:
                pass  # best-effort fusion; fallback to computed values

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            # NOTE: src_idx is (src in x), dst_idx is (dst in x), unm_idx is (unm in x)
            out.scatter_(dim=-2, index=dst_idx.expand(B, reduce_num, c), src=dst)
            out.scatter_(dim=-2, index=unm_idx.expand(B, tu, c), src=unm)
            out.scatter_(dim=-2, index=src_idx.expand(B, reduce_num*dim_index, c), src=src)
            return out

    return merge, mprune, unmerge

# NOTE: Fake IDM
# Since xFormers does not support the explicit output of attention maps, FIDM removes the dependency on attention maps.
# Instead, within each window, we select the tokens with the highest frequency priority scores as the "attentive group", while the remaining tokens are assigned to the "inattentive group".
# This design enables effective separation without relying on attention outputs.
def FIDM(
        metric: torch.Tensor,
        reduce_num: int = 0,
        window_size: Tuple[int, int] = (2,2),
        no_rand: bool = False,
        generator: torch.Generator = None,
        tore_info: Dict = None
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
        sx, sy = int(window_size[0]), int(window_size[1])

        h = int(math.sqrt(N))
        w = h
        assert h * w == N and h % sy == 0 and w % sx == 0
        hsy, wsx =  h // sy, w // sx

        # Decide dst inside each (sy, sx) window: prefer max last_independent if available else random
        use_li = (
            tore_info is not None and "states" in tore_info and
            tore_info["states"].get("last_independent") is not None and
            tore_info["states"]["last_independent"].shape[1] == N
        )

        if use_li:
            li = tore_info["states"]["last_independent"]  # [B, N]
            li_grid = li.view(B, h, w)
            # Reshape to windows: [B, hsy, sy, wsx, sx] -> permute to [B, hsy, wsx, sy, sx]
            li_windows = li_grid.view(B, hsy, sy, wsx, sx).permute(0, 1, 3, 2, 4)
            # Flatten each window tokens: [B, hsy, wsx, sy*sx]
            li_flat_win = li_windows.reshape(B, hsy, wsx, sy * sx)
            # Argmax per window: returns index in [0, sy*sx)
            dst_pos = li_flat_win.argmax(dim=-1, keepdim=True)  # [B, hsy, wsx, 1]
        else:
            # Random fallback (per batch) to maintain original behavior when no last_independent
            if no_rand:
                dst_pos = torch.zeros(B, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
            else:
                if generator is not None:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device, generator=generator)
                else:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device)

        # Build sentinel buffer: -1 at dst position, 0 elsewhere
        idx_buffer_view = torch.zeros(B, hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        neg_one = -torch.ones_like(dst_pos, dtype=idx_buffer_view.dtype)
        idx_buffer_view.scatter_(dim=3, index=dst_pos, src=neg_one)

        # Reshape to spatial (with same ordering as original single-batch code)
        # Original single batch: (hsy, wsx, sy, sx).transpose(1,2) -> (hsy, sy, wsx, sx)
        # Multi-batch adaption:
        idx_buffer_view = idx_buffer_view.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # Argsort per batch to obtain dst|src partition indices
        rand_idx = idx_buffer.reshape(B, -1, 1).argsort(dim=1)

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
        reduce_num = reduce_num // 16 * 16 # ensure multiple of 16

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., reduce_num:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :reduce_num, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # Expand original dst/src index tensors along batch for downstream mapping
        a_idx_b = a_idx.expand(B, a_idx.shape[1], 1)  # [B, N_src, 1]
        b_idx_b = b_idx.expand(B, b_idx.shape[1], 1)  # [B, num_dst, 1]

        # Store raw token indices (no channel expansion yet)
        dst_in_x_index = b_idx_b                       # [B, num_dst, 1]
        unm_in_x_index = gather(a_idx_b, dim=1, index=unm_idx)  # [B, a_len-reduce_num, 1]
        src_in_x_index = gather(a_idx_b, dim=1, index=src_idx)  # [B, reduce_num, 1]

        independent_idx = None
        unindependent_idx = None

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_in_x_index.squeeze(-1), dst_in_x_index.squeeze(-1)], dim=-1)
            unindependent_idx = src_in_x_index.squeeze(-1)
        else:
            independent_idx = unm_in_x_index.squeeze(-1)
            unindependent_idx = torch.cat([src_in_x_index.squeeze(-1), dst_in_x_index.squeeze(-1)], dim=-1)
        update_last_independent(independent_idx)

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

            # Optional fusion with cached features for dst/src tokens (unm remains as-is)
            try:
                # Determine cache_name directly from phase state; align with features dict keys
                cache_name = tore_info.get("states", {}).get("unmerge_phase", None)
                if cache_name not in ("attn_output", "attn_output2", "mlp_output"):
                    pass
                mcw = float(tore_info.get("args", {}).get("mcw", 1.0))
                l = tore_info.get("states", {}).get("layer_current", -1)
                key = f"l{l}"
                cache_dict = tore_info.get("features", {}).get(cache_name, None)
                cache_full = cache_dict.get(key) if isinstance(cache_dict, dict) else None
                if cache_full is not None:
                    cache_full = cache_full.to(device=x.device, dtype=x.dtype)
                    # Gather cached values by original positions
                    cached_dst = gather(cache_full, dim=1, index=dst_in_x_index.expand(B, dst_in_x_index.shape[1], c))
                    cached_src = gather(cache_full, dim=1, index=src_in_x_index.expand(B, src_in_x_index.shape[1], c))
                    # fuse
                    dst = mcw * dst + (1.0 - mcw) * cached_dst
                    src = mcw * src + (1.0 - mcw) * cached_src
            except Exception:
                pass  # best-effort fusion; fallback to computed values

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            # NOTE: a_idx is (a in x) b_idx is (dst in x), 
            # NOTE: dst_idx is (src in dst), unm_idx is (unm in a), (src_idx) is (src in a)

            out.scatter_(dim=-2, index=dst_in_x_index.expand(B, dst_in_x_index.shape[1], c), src=dst)
            out.scatter_(dim=-2, index=unm_in_x_index.expand(B, unm_in_x_index.shape[1], c), src=unm)
            out.scatter_(dim=-2, index=src_in_x_index.expand(B, src_in_x_index.shape[1], c), src=src)
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

def compute_ratio(tore_info: Dict[str, Any]) -> float:
    """Compute ratio_current using values stored in `tore_info`.

    The function extracts the following fields from tore_info:
      - ratio, deviation
      - step_current, step_count
      - layer_current, layer_count
      - protect_steps_frequency, protect_layers_frequency

    If a protect frequency is negative (e.g. -1) it is treated as disabled.
    Returns 0.0 when current step or layer is protected; otherwise returns linear
    interpolation from ratio+deviation (step=0) to ratio-deviation (step=step_count-1).
    """
    args = tore_info.get("args", {})
    states = tore_info.get("states", {})

    ratio = args.get("ratio", 0.5)
    deviation = args.get("deviation", 0.2)
    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 1)
    layer_current = states.get("layer_current", 0)
    layer_count = states.get("layer_count", 1)
    # NOTE: protect_steps_frequency no longer affects ratio scheduling. It is now
    # handled inside the transformer block forward pass to completely bypass
    # merge/unmerge logic for protected steps instead of forcing ratio=0 here.
    protect_steps_frequency = args.get("protect_steps_frequency", None)  # kept for backward compat, unused below
    protect_layers_frequency = args.get("protect_layers_frequency", None)

    def is_protected(idx, total, freq):
        # Treat None or negative frequency as disabled
        if freq is None or freq < 0:
            return False
        # frequency == 0 is invalid -> treat as disabled
        if freq == 0:
            return False
        if idx % freq == 0:
            return True
        if idx == max(total - 1, 0):
            return True
        return False

    # Only layer protection can still zero out ratio here; step protection is handled in block forward.
    if is_protected(layer_current, layer_count, protect_layers_frequency):
        tore_info["states"]["last_independent"] = None
        return 0.0

    if step_count > 1:
        progress = step_current / (step_count - 1)
    else:
        progress = 0.0

    # Use a cosine-shaped descent to go from (ratio + deviation) -> (ratio - deviation).
    # We map progress in [0,1] to alpha = cos(progress * pi/2), which goes 1 -> 0.
    # Then interpolate: value = (ratio - deviation) + 2*deviation*alpha
    alpha = math.cos(progress * math.pi / 2)
    ratio_current = (ratio - deviation) + (2.0 * deviation) * alpha

    return float(ratio_current)

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a SDTM Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tore_info"):
            for hook in module._tore_info["hooks"]:
                hook.remove()
            module._tore_info["hooks"].clear()

        if module.__class__.__name__ == "SDTMBlock":
            module.__class__ = module._parent
    
    return model

def compute_merge(x: torch.Tensor, tore_info: Dict[str, Any]) -> Tuple[Callable, ...]:

    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"

    # 使用模块级 compute_ratio
    ratio_current = compute_ratio(tore_info)
    tore_info["states"]["ratio_current"] = ratio_current

    reduce_num = int(x.shape[1] * ratio_current)
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing, do_nothing

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if tore_info["args"]["generator"] is None:
        tore_info["args"]["generator"] = init_generator(x.device)
    elif tore_info["args"]["generator"].device != x.device:
        tore_info["args"]["generator"] = init_generator(x.device, fallback=tore_info["args"]["generator"])

    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else tore_info["args"]["use_rand"]

    # Choose strategy based on switch_step: SSM when step_current <= switch_step, otherwise IDM
    step_current = tore_info["states"].get("step_current", 0)
    switch_step = tore_info["args"].get("switch_step", 20)
    if step_current <= switch_step:
        m, pm, u  = SSM(metric=x, reduce_num=reduce_num, window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),  
                        no_rand=not use_rand, generator=tore_info["args"]["generator"], tore_info=tore_info)
    else:
        m, pm, u  = FIDM(metric=x, reduce_num=reduce_num, window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),  
                        no_rand=not use_rand, generator=tore_info["args"]["generator"], tore_info=tore_info)

    if tore_info["args"]["pseudo_merge"]:
        m_a, u_a = (pm, u) if tore_info["states"]["merge_attn"]==True else (do_nothing, do_nothing)
        m_m, u_m = (pm, u) if tore_info["states"]["merge_mlp"]==True  else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if tore_info["states"]["merge_attn"]==True else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if tore_info["states"]["merge_mlp"]==True  else (do_nothing, do_nothing)

    return m_a, m_m, u_a, u_m

def make_SDTM_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class StableDiffusion3Pipeline_SDTM(pipe_class):
        # Save for unpatching later
        _parent = pipe_class

        def __call__(self, *args, **kwargs):
            self._tore_info["states"]["step_count"] = kwargs['num_inference_steps']
            self._tore_info["states"]["step_iter"] = list(range(kwargs['num_inference_steps']))
            self._tore_info["states"]["last_independent"] = None
            output = super().__call__(*args, **kwargs)
            return output

    return StableDiffusion3Pipeline_SDTM

def make_SDTM_model(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    
    class SD3Transformer2DModel_SDTM(model_class):
        _parent = model_class

        def forward(self, *args, **kwargs):
            self._tore_info["states"]["layer_count"] = self.config.num_layers
            # pop next step; wrapper installation is handled once in apply_SDTM to avoid per-forward overhead
            self._tore_info["states"]["step_current"] = self._tore_info["states"]["step_iter"].pop(0)
            self._tore_info["states"]["layer_iter"] = list(range(self.config.num_layers))
            if self._tore_info["states"]["step_current"] <= self._tore_info["args"]["switch_step"]:
                self._tore_info["states"]["merge_attn"] = True
                self._tore_info["states"]["merge_mlp"] = True
            else:
                self._tore_info["states"]["merge_attn"] = False
                self._tore_info["states"]["merge_mlp"] = True
            output = super().forward(*args, **kwargs)
            return output

    return SD3Transformer2DModel_SDTM

def make_SDTM_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class JointTransformerBlock_SDTM(block_class):
        
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

            # Determine if current step is protected (skip merge/unmerge entirely)
            args = self._tore_info.get("args", {})
            states = self._tore_info.get("states", {})
            protect_freq = args.get("protect_steps_frequency", None)
            step_current = states.get("step_current", 0)

            def _is_protected_step(idx, freq):
                if freq is None or freq < 0 or freq == 0:
                    return False
                # protect steps that are multiples of freq and the final step handled at pipeline level
                return idx % freq == 0

            protected_step = _is_protected_step(step_current, protect_freq)

            # helper: store intermediate features by (step, layer), device configurable
            def _store_feature(name: str, tensor: torch.Tensor):
                try:
                    feat = self._tore_info.setdefault("features", {})
                    if not isinstance(feat.get(name), dict):
                        feat[name] = {}
                    l = self._tore_info.get("states", {}).get("layer_current", -1)
                    key = f"l{l}"
                    feat[name][key] = tensor.detach()
                except Exception:
                    # best-effort; do not break the forward pass if logging fails
                    pass

            if protected_step:
                attn_output, context_attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    **joint_attention_kwargs,
                )
                attn_output = gate_msa.unsqueeze(1) * attn_output
                _store_feature("attn_output", attn_output)
                
                hidden_states = hidden_states + attn_output

                if self.use_dual_attention:
                    attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
                    attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                    _store_feature("attn_output2", attn_output2)
                    hidden_states = hidden_states + attn_output2

                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)
                ff_output = gate_mlp.unsqueeze(1) * ff_output
                _store_feature("mlp_output", ff_output)
                hidden_states = hidden_states + ff_output
            else:
                #! Step 1_1: Compute_Merge
                m_a, m_m, u_a, u_m = compute_merge(norm_hidden_states, self._tore_info)
                if self.use_dual_attention:
                    m_a2, _, u_a2, _ = compute_merge(norm_hidden_states2, self._tore_info)

                #! Step 1_2_1: Merge_Attn
                norm_hidden_states = m_a(norm_hidden_states)
                attn_output, context_attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    **joint_attention_kwargs,
                )
                attn_output = gate_msa.unsqueeze(1) * attn_output
                #! Step 1_2_2: UnMerge_Attn
                self._tore_info["states"]["unmerge_phase"] = "attn_output"
                attn_output = u_a(attn_output)
                if self._tore_info["args"]["cache_each_step"]==True: _store_feature("attn_output", attn_output)
                hidden_states = hidden_states + attn_output

                if self.use_dual_attention:
                    #! Step 1_2_3: Merge_DualAttn
                    norm_hidden_states2 = m_a2(norm_hidden_states2)
                    attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
                    attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                    #! Step 1_2_4: UnMerge_DualAttn
                    self._tore_info["states"]["unmerge_phase"] = "attn_output2"
                    attn_output2 = u_a2(attn_output2)
                    if self._tore_info["args"]["cache_each_step"]==True: _store_feature("attn_output2", attn_output2)
                    hidden_states = hidden_states + attn_output2

                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                #! Step 1_3_1: Merge_MLP
                norm_hidden_states = m_m(norm_hidden_states)
                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)
                ff_output = gate_mlp.unsqueeze(1) * ff_output
                #! Step 1_3_2: UnMerge_MLP
                self._tore_info["states"]["unmerge_phase"] = "mlp_output"
                ff_output = u_m(ff_output)
                if self._tore_info["args"]["cache_each_step"]==True:  _store_feature("mlp_output", ff_output)
                hidden_states = hidden_states + ff_output

                self._tore_info["states"]["unmerge_phase"] = None
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
    
    return JointTransformerBlock_SDTM

def apply_SDTM(
    pipe: torch.nn.Module,
    ratio: float = 0.5,
    deviation: float = 0.2,
    switch_step: int = 20,
    use_rand: bool = True,
    sx: int = 2,
    sy: int = 2,
    a_s: float = 0.05,
    a_d: float = 0.05,
    a_p: float = 2,
    pseudo_merge: bool = False,
    mcw: float = 0.2,
    protect_steps_frequency: int = None,
    protect_layers_frequency: int = None,
    merge_attn: bool = False,
    merge_mlp: bool = False,
):

    # Make sure the module is not currently patched
    remove_patch(pipe)
    make_pipe_fn = make_SDTM_pipe
    pipe.__class__ = make_pipe_fn(pipe.__class__)

    pipe._tore_info = {
        "type": "SDTM",
        "args": {
            "ratio": ratio,
            "deviation": deviation,
            "switch_step": switch_step,
            "use_rand": use_rand,
            "sx": sx,
            "sy": sy,
            "a_s": a_s,
            "a_d": a_d,
            "a_p": a_p,
            "pseudo_merge": pseudo_merge,
            "mcw": mcw,
            "protect_steps_frequency": protect_steps_frequency,
            "protect_layers_frequency": protect_layers_frequency,
            "generator": None,
            "cache_each_step": False,
        },
        "features": {
            "attn_output": None,
            "attn_output2": None,
            "mlp_output": None,
        },
        "states": {
            "last_independent": None,
            "ratio_current": ratio,
            "step_count": None,
            "step_iter": None,
            "step_current": None,
            "layer_count": None,
            "layer_iter": None,
            "layer_current": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
            "unmerge_phase": None,
        }
    }

    model = pipe.transformer
    make_model_fn = make_SDTM_model
    model.__class__ = make_model_fn(model.__class__)
    model._tore_info = pipe._tore_info
    for _, module in model.named_modules():
        if isinstance_str(module, "JointTransformerBlock"):
            make_block_fn = make_SDTM_block
            module.__class__ = make_block_fn(module.__class__)
            module._tore_info = pipe._tore_info
            # Disable dual attention on patched blocks to simplify behavior
            try:
                module.use_dual_attention = False
            except Exception:
                pass
    # Note: attention map collection is handled by _call_attn_with_get_scores so no global monkeypatch is needed.
    return pipe
