import math
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
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = math.prod(new_size) + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.pow(num_pos_tokens - num_prefix_tokens, 1/3))
        old_size = hw, hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, *old_size, -1).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def resample_abs_pos_embed_nhwc(
        posemb,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    if (
        new_size[0] == posemb.shape[-4] 
        and new_size[1] == posemb.shape[-3]
        and new_size[2] == posemb.shape[-2]
    ):
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    # do the interpolation
    posemb = posemb.reshape(
        1, 
        posemb.shape[-4],
        posemb.shape[-3], 
        posemb.shape[-2], 
        posemb.shape[-1]
    ).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 4, 1).to(orig_dtype)

    return posemb


from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
