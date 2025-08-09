import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SigLIPLoss(nn.Module):
    def __init__(
        self, 
        learnable_t: bool = True, 
        learnable_b: bool = True, 
        normalize: bool = True,
        init_t: float = math.log(10),  # Default temperature for SigLIP
        init_b: float = -10.0,  # Default bias for SigLIP
    ):
        """
        SigLip loss for aligning image and text embeddings.

        Args:
            learnable_t (bool): If True, temperature `t` is a learnable parameter.
            learnable_b (bool): If True, bias `b` is a learnable parameter.
            normalize (bool): If True, embeddings are L2-normalized before computing logits.
            init_t (float): Initial value for temperature.
            init_b (float): Initial value for bias.
        """
        super().__init__()
        self.normalize = normalize

        # Define learnable parameters for temperature and bias
        self.t = nn.Parameter(torch.tensor(init_t), requires_grad=learnable_t)
        self.b = nn.Parameter(torch.tensor(init_b), requires_grad=learnable_b)

    @staticmethod
    def slice_loglik(logits: torch.Tensor, include_pos: bool) -> torch.Tensor:
        """
        Computes the log-likelihood for positive and negative pairs in the logits matrix.

        Args:
            logits (torch.Tensor): Logits matrix of shape (batch_size, batch_size).
            include_pos (bool): If True, includes positive pairs in the log-likelihood calculation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Positive and negative log-likelihoods.
        """
        B = logits.size(0)

        if include_pos:
            # Positive pairs are on the diagonal
            pos_mask = torch.eye(B, device=logits.device)
        else:
            # If not including positive pairs, we treat all pairs as negatives
            pos_mask = torch.zeros(B, B, device=logits.device)
        neg_mask = 1.0 - pos_mask

        m1 = -torch.ones_like(logits, device=logits.device)
        m1 += 2 * pos_mask  # Add identity matrix to the diagonal

        # joint log-likelihood
        loglik = F.logsigmoid(m1 * logits)

        pos_ll = (loglik * pos_mask).sum(dim=-1)  # positive log likelihood
        neg_ll = (loglik * neg_mask).sum(dim=-1)  # negative log likelihood

        return pos_ll, neg_ll

    def forward(
        self, 
        zimg: torch.Tensor, 
        ztxt: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Computes the alignment loss between image and text embeddings.

        Args:
            zimg (torch.Tensor): Image embeddings of shape (batch_size, embedding_dim).
            ztxt (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Computed loss value.
        """
        if self.normalize:
            zimg = F.normalize(zimg, p=2, dim=-1)
            ztxt = F.normalize(ztxt, p=2, dim=-1)

        # ---- setup distributed ----
        if not dist.is_initialized():
            # fallback to single-GPU
            logits = zimg @ ztxt.t()
            logits = logits * torch.exp(self.t) + self.b

            pos_ll, neg_ll = self.slice_loglik(logits, include_pos=True)

            pos_loss = -pos_ll.mean()  # mean loss for positives
            neg_loss = -neg_ll.mean()  # mean loss for negatives

            loss = pos_loss + neg_loss  # total loss
            
            if return_details:
                return loss, {"pos_loss": pos_loss.item(), "neg_loss": neg_loss.item()}
            return loss

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # accumulators (sum of per-sample losses)
        pos_sum = torch.tensor(0., device=zimg.device, dtype=zimg.dtype)
        neg_sum = torch.tensor(0., device=zimg.device, dtype=zimg.dtype)

        # total number of samples across all ranks
        B = torch.tensor(zimg.size(0), device=zimg.device, dtype=zimg.dtype)
        dist.all_reduce(B, op=dist.ReduceOp.SUM)

        for k in range(world_size):
            # buffer for the rotating text embeddings
            # start by copying the local ztxt into it
            ztxt_rot = ztxt.clone()
            dist.barrier()  # ensure all ranks are ready cloning
            dist.broadcast(ztxt_rot, src=k)

            # now compute this “slice” of the full N×N logits:
            logits = zimg @ ztxt_rot.t()  # (batch_size, batch_size)
            logits = logits * torch.exp(self.t) + self.b

            if k == rank:
                pos_ll, neg_ll = self.slice_loglik(logits, include_pos=True)
                pos_ll, neg_ll = pos_ll / B, neg_ll / B  # normalize by batch size
                pos_sum += pos_ll.sum()  # accumulate positive log likelihood
                neg_sum += neg_ll.sum()  # accumulate negative log likelihood

            else:
                # for all other slices, we only compute the negative log likelihood
                # since the positive pairs are already included in the first slice
                pos_ll, neg_ll = self.slice_loglik(logits, include_pos=False)
                neg_ll = neg_ll / B  # normalize by batch size
                neg_sum += neg_ll.sum()

        # add across devices to get the mean
        # we already divided by batch size before
        dist.all_reduce(pos_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_sum, op=dist.ReduceOp.SUM)

        # Compute the final loss
        pos_loss = -pos_sum
        neg_loss = -neg_sum

        # `total_loss` is now the total SigLIP loss summed over all examples and all ranks
        total_loss = pos_loss + neg_loss

        if return_details:
            return total_loss, {"pos_loss": pos_loss.item(), "neg_loss": neg_loss.item()}
        return total_loss
