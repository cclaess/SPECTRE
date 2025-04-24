import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class iBOTPatchLoss(nn.Module):
    """
    Patch-level contrastive loss used in iBOT.

    This loss function aligns patch representations from a teacher-student model,
    preventing collapse and ensuring stable training through centering.
    """

    def __init__(
        self,
        output_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the iBOT patch loss.

        Args:
            output_dim: Output dimensionality of patch embeddings.
            student_temp: Temperature parameter for the student model.
            center_momentum: Momentum for updating the teacher center.
            warmup_teacher_temp: Initial temperature for the teacher.
            teacher_temp: Final temperature for the teacher.
            warmup_teacher_temp_epochs: Epochs over which to warm up the teacher temperature.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.register_buffer("center", torch.zeros(1, 1, output_dim))

        # Warmup schedule for teacher temperature
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Computes the cross-entropy loss for patch tokens.

        Args:
            teacher_out: (B, N, D) Tensor of teacher patch embeddings.
            student_out: (B, N, D) Tensor of student patch embeddings.
            mask: (B, N) Binary mask tensor for valid patches.
            epoch: The current training epoch.

        Returns:
            Scalar loss value.
        """
        teacher_temp = (
            self.teacher_temp_schedule[epoch]
            if epoch < self.warmup_teacher_temp_epochs
            else self.teacher_temp
        )

        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        student_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        loss = -torch.sum(teacher_out * student_out, dim=-1)
        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        loss = loss.mean()

        self.update_center(teacher_out)
        return loss

    def forward_masked(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
        n_masked_patches: int = None,
        masks_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes the cross-entropy loss for masked patch tokens.

        Args:
            teacher_out: (B, N, D) Tensor of teacher patch embeddings.
            student_out: (B, N, D) Tensor of student patch embeddings.
            mask: (B, N) Binary mask tensor for valid patches.
            epoch: The current training epoch.
            n_masked_patches: Optional number of masked patches to consider.
            masks_weight: Optional tensor for weighting masked patches.

        Returns:
            Scalar loss value.
        """
        teacher_temp = (
            self.teacher_temp_schedule[epoch]
            if epoch < self.warmup_teacher_temp_epochs
            else self.teacher_temp
        )
        
        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        loss = torch.sum(teacher_out * F.log_softmax(student_out / self.student_temp, dim=-1), dim=-1)

        if masks_weight is None:
            masks_weight = (
                (1 / mask.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(mask)[mask]
            )

        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]

        loss = loss * masks_weight
        loss = -loss.sum() / mask.shape[0]

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Moving average update of the center used for the teacher output."""
        batch_center = teacher_out.mean(dim=(0, 1), keepdim=True)
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
