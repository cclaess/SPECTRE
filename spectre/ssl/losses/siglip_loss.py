import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.t = nn.Parameter(torch.tensor(init_t)) if learnable_t else init_t
        self.b = nn.Parameter(torch.tensor(init_b)) if learnable_b else init_b

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
            zimg = F.normalize(zimg, p=2, dim=-1)  # (batch_size, embedding_dim)
            ztxt = F.normalize(ztxt, p=2, dim=-1)  # (batch_size, embedding_dim)

        logits = torch.matmul(zimg, ztxt.T)  # Compute similarity matrix (batch_size, batch_size)
        logits = logits * self.t + self.b  # Apply scaling and bias

        batch_size = logits.size(0)
        eye = torch.eye(batch_size, device=logits.device)  # Identity matrix for positives

        # m1_diag1 = -torch.ones_like(logits, device=logits.device) + 2 * eye
        # loglik = F.logsigmoid(m1_diag1 * logits)  # Log sigmoid  (batch_size, batch_size)
        # nll = -torch.sum(loglik, dim=-1)  # Negative log likelihood
        # loss = torch.mean(nll)  # Final mean loss

        # if not return_details:
        #     return loss
        
        # pos_loglik = torch.diag(loglik)
        # pos_loss = -pos_loglik.sum() / batch_size  # Contribution of positive pairs

        # neg_loglik = loglik * (1 - eye)  # Exclude diagonal for negatives
        # neg_loss = -neg_loglik.sum() / batch_size  # Contribution of negative pairs

        # return loss, {"pos_loss": pos_loss.item(), "neg_loss": neg_loss.item()}

        pos_loglik = torch.diag(F.logsigmoid(logits))

        print("pos_loglik: ", pos_loglik.shape, pos_loglik)

        neg_loglik = F.logsigmoid(-logits) * (1 - eye)
        neg_loglik_avg = neg_loglik.sum(dim=-1) / (batch_size - 1)  # Average per sample

        print("neg_loglik_avg: ", neg_loglik_avg)
        
        loss_per_sample = -pos_loglik - neg_loglik_avg
        loss = loss_per_sample.mean()

        if not return_details:
            return loss
        
        pos_loss = -pos_loglik.sum() / batch_size  # Contribution of positive pairs
        neg_loss = -neg_loglik_avg.sum() / batch_size  # Contribution of negative pairs

        return loss, {"pos_loss": pos_loss.item(), "neg_loss": neg_loss.item()}
