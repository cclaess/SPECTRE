import os
import argparse

from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import DINO
from spectre.ssl.losses import DINOLoss
from spectre.configs import default_config_dino
from spectre.utils.config import setup
from spectre.utils.models import update_momentum
from spectre.utils.dataloader import get_dataloader
from spectre.utils.scheduler import CosineWarmupScheduler, cosine_schedule


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain DINO")
    parser.add_argument("--config_file", type=str, default="configs/dino_default.yaml", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="command line arguments to override config file")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory to save checkpoints and logs")
    return parser


def main(cfg):
    """
    Main function to run pretraining.
    """
    print(cfg)

    # Initialize accelerator
    accelerator = Accelerator(
        log_with="wandb" if cfg.train.log_wandb else None,
    )

    # Initialize wandb
    if cfg.train.log_wandb:
        accelerator.init_trackers(
            project_name="spectre",
            config=vars(cfg),
            init_kwargs={
                "name": "dino-pretrain-" + cfg.model.architecture,
                "dir": cfg.train.output_dir / "logs",
            },
        )
    
    # Get dataloader
    data_Loader = get_dataloader(
        cfg.train.dataset,
        cfg.train.dataset_path,
        include_reports=False,
        cache_dataset=cfg.train.cache_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    # Initialize backbone
    if cfg.model.architecture in models.__dict__ and cfg.model.architecture.startswith("vit"):
        backbone = models.__dict__[cfg.model.architecture]()
        embed_dim = backbone.embed_dim
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    # Initialize DINO model
    model = DINO(
        backbone, 
        input_dim=embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        output_dim=cfg.model.output_dim,
    )

    # Initialize criterion
    criterion = DINOLoss(
        output_dim=cfg.model.output_dim,
        warmup_teacher_temp=cfg.loss.warmup_teacher_temp,
        teacher_temp=cfg.loss.teacher_temp,
        warmup_teacher_temp_epochs=cfg.loss.warmup_teacher_temp_epochs,
        student_temp=cfg.loss.student_temp,
        center_momentum=cfg.loss.center_momentum,
    )

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
    )

    # calculate number of steps for training
    num_steps = cfg.optim.epochs * len(data_Loader) // accelerator.num_processes
    num_warmup_steps = cfg.optim.warmup_epochs * len(data_Loader) // accelerator.num_processes

    # Initialize learning rate scheduler
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=num_warmup_steps,
        max_epochs=num_steps,
        start_value=cfg.optim.lr,
        end_value=cfg.optim.min_lr,
    )

    # Prepare model, data, and optimizer for training
    model, data_Loader, optimizer, lr_scheduler = accelerator.prepare(
        model, data_Loader, optimizer, lr_scheduler,
    )

    # Start training
    global_step: int = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        for batch in data_Loader:
            optimizer.zero_grad()

            # Update learning rate
            lr_scheduler.step()
            
            # Update momentum
            momentum = cosine_schedule(
                global_step, num_steps, cfg.model.momentum_teacher, cfg.model.momentum_teacher_end
            )
            update_momentum(model.student_backbone, model.teacher_backbone, momentum)
            update_momentum(model.student_head, model.teacher_head, momentum)

            # Update weight decay
            weight_decay = cosine_schedule(
                global_step, num_steps, cfg.optim.weight_decay, cfg.optim.weight_decay_end
            )
            optimizer.param_groups[0]["weight_decay"] = weight_decay
            
            # Forward pass
            global_views = [batch["image"].view(-1, 2, *batch["image"].shape[2:])[:, i] for i in range(2)]
            local_views = [batch["image_local"].view(-1, 8, *batch["image_local"].shape[2:])[:, i] for i in range(8)]

            teacher_outputs = [model.forward_teacher(view) for view in global_views]
            student_outputs = [model(view) for view in global_views + local_views]

            loss = criterion(teacher_outputs, student_outputs, epoch=epoch)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update model
            if cfg.train.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(
                    model.student_backbone.named_parameters(), cfg.train.clip_grad_norm
                )
                accelerator.clip_grad_norm_(
                    model.student_head.named_parameters(), cfg.train.clip_grad_norm
                )
            model.student_head.cancel_last_layer_gradients(epoch)
            optimizer.step()

            global_step += 1
        
        if (epoch + 1) % cfg.train.saveckp_freq == 0:
            accelerator.save_model(
                model, 
                os.path.join(cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}.pt")
            )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = setup(args, default_config_dino)
    main(cfg)
    