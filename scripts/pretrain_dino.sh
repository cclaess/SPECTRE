cd "$(dirname "$(dirname "$0")")" | exit 1
export $(cat .env | xargs)

wandb login

accelerate launch --config_file spectre/configs/accelerate_default.yaml \
    experiments/pretraining/pretrain_dino.py \
        --config_file spectre/configs/dino_default.yaml \
        --output_dir outputs/pretraining/dino