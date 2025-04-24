import math
from enum import Enum
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.

    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False


def activate_requires_grad(model: nn.Module):
    """Activates the requires_grad flag for all parameters of a model.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = True


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component fo models such as MoCo or BYOL.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def repeat_token(token: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Repeats a token size times.

    Args:
        token: Token tensor with shape (1, 1, dim).
        size: (batch_size, sequence_length) tuple.

    Returns:
        Tensor with shape (batch_size, sequence_length, dim) containing copies
        of the input token.
    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)


def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Expands the index along the last dimension of the input tokens.

    Args:
        index:
            Index tensor with shape (batch_size, idx_length) where each entry is
            an index in [0, sequence_length).
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).

    Returns:
        Index tensor with shape (batch_size, idx_length, dim) where the original
        indices are repeated dim times along the last dimension.
    """
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index


def get_at_index(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects tokens at index.

    Args:
        tokens:
            Token tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length) where each entry is
            an index in [0, sequence_length).

    Returns:
        Token tensor with shape (batch_size, index_length, dim) containing the
        selected tokens.
    """
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Copies all values into the input tensor at the given indices.

    Args:
        tokens: Tokens tensor with shape (batch_size, sequence_length, dim).
        index: Index tensor with shape (batch_size, index_length).
        value: Value tensor with shape (batch_size, index_length, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.
    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def mask_at_index(
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Copies mask token into the input tensor at the given indices.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        mask_token:
            Value tensor with shape (1, 1, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)
    return (1 - mask) * tokens + mask * mask_token


def mask_bool(tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch. Tensor:
    """Returns a tensor with tokens replaced by the mask tokens in all positions where
    the mask is True.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        mask:
            Boolean mask tensor with shape (batch_size, sequence_length).
        mask_token:
            Mask token with shape (1, 1, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) where tokens[i, j]
        is replaced by the mask token if mask[i, j] is True.
    """
    # Convert to int for multiplication.
    mask = mask.unsqueeze(-1).to(torch.bool).to(torch.int)
    return (1 - mask) * tokens + mask * mask_token


def patchify(images: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """Converts a batch of input images into patches.

    Args:
        images:
            Images tensor with shape (batch_size, channels, height, width, depth)
        patch_size:
            Patch size in pixels. Image width and height must be multiples of
            the patch size.

    Returns:
        Patches tensor with shape (batch_size, num_patches, channels * math.prod(patch_size))
        where num_patches = image_width / patch_size * image_height / patch_size.

    """
    N, C, H, W, D = images.shape
    assert (
        H % patch_size[0] == 0
        and W % patch_size[1] == 0
        and D % patch_size[2] == 0
    ), "Image height, width, and depth must be multiples of the patch size."

    patch_h =  H // patch_size[0]
    patch_w =  W // patch_size[1]
    patch_d =  D // patch_size[2]

    num_patches = patch_h * patch_w * patch_d
    patches = images.reshape(shape=(
        N, C, 
        patch_h, patch_size[0], 
        patch_w, patch_size[1], 
        patch_d, patch_size[2],
    ))
    patches = torch.einsum("nchpwqdr->nhwdpqrc", patches)
    patches = patches.reshape(shape=(N, num_patches, math.prod(patch_size) * C))
    return patches


def random_token_mask(
    size: Tuple[int, int],
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Creates random token masks.

    Args:
        size:
            Size of the token batch for which to generate masks.
            Should be (batch_size, sequence_length).
        mask_ratio:
            Percentage of tokens to mask.
        mask_class_token:
            If False the class token is never masked. If True the class token
            might be masked.
        device:
            Device on which to create the index masks.

    Returns:
        A (index_keep, index_mask) tuple where each index is a tensor.
        index_keep contains the indices of the unmasked tokens and has shape
        (batch_size, num_keep). index_mask contains the indices of the masked
        tokens and has shape (batch_size, sequence_length - num_keep).
        num_keep is equal to sequence_length * (1- mask_ratio).

    """
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        # make sure that class token is not masked
        noise[:, 0] = -1
        num_keep = max(1, num_keep)

    # get indices of tokens to keep
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]

    return idx_keep, idx_mask


def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: List[int],
        num_prefix_tokens: int = 1,
        interpolation: str = 'trilinear',
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] * new_size[2] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], -1).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def resample_abs_pos_embed_nhwdc(
        posemb: torch.Tensor,
        new_size: List[int],
        interpolation: str = 'trilinear',
):
    if new_size[0] == posemb.shape[-4] and new_size[1] == posemb.shape[-3] and new_size[2] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    posemb = posemb.reshape(1, posemb.shape[-4], posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).to(orig_dtype)

    return posemb


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'trilinear',
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple[int, int, int]): target shape (depth, height, width).
        interpolation (str): interpolation for resize
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 5, "Five dimensions expected"
    assert len(new_size) == 3, "New shape should only be (height, width, depth)"
    old_size = patch_embed.shape[-3:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    """ Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forward() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        assert 0 < indices <= num_features, f'last-n ({indices}) is out of range (1 to {num_features})'
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            assert 0 <= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})'
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


class Format(str, Enum):
    NCHWD = 'NCHWD'
    NHWDC = 'NHWDC'
    NCL = 'NCL'
    NLC = 'NLC'


def nchwd_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWDC:
        x = x.permute(0, 2, 3, 4, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwdc_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NCHWD:
        x = x.permute(0, 4, 1, 2, 3)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x