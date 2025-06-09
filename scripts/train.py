from src.train.trainer import Trainer

if __name__ == '__main__':
    # input_image_size在dataset以及anchor生成都需要使用，所以进行统一处理
    cfg_trainer = {
        'data_dir': r'D:\Data\deeplearning\datasets\Anti-UAV\val',
        'num_epochs': 3,
        'batch_size': 32,
        'input_image_size': [224, 224],
        'num_anchor_per_pixel': 3,
        'anchor_ratios_per_level': [[4, 8, 16], [32, 48, 64], [80, 96, 128]]
    }

    trainer = Trainer(cfg_trainer=cfg_trainer)
    trainer.train()
