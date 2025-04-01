import random
import torch
from pytorch3dunet.unet3d.config import load_config, copy_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.utils import get_train_loaders

logger = get_logger('TrainingSetup')

def main():
    config, config_path = load_config()
    logger.info(config)
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True

    # 调试 DataLoader
    loaders = get_train_loaders(config)
    train_loader = loaders['train']
    logger.info(f"Train loader batch size: {train_loader.batch_size}")
    logger.info(f"Training dataset size: {len(train_loader.dataset)} patches")
    for batch in train_loader:
        raw, label = batch
        logger.info(f"Batch raw shape: {raw.shape}")
        logger.info(f"Batch label shape: {label.shape}")
        break

    trainer = create_trainer(config)
    copy_config(config, config_path)
    trainer.fit()

if __name__ == '__main__':
    main()