import os
from gorilla.config import Config
from utils import *
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config.fromfile(args.config)
    
logger = IOStream(os.path.join(cfg.log_dir, "run.log"))

if __name__ == "__main__":
    if cfg.get("seed") is not None:
        set_random_seed(cfg.seed)
        logger.cprint("Set seed to %d" % cfg.seed)
    model = build_model(cfg)
    model = model.to("cuda")
    model = torch.nn.DataParallel(model)
    
    logger.cprint("Training from scratch!")

    dataset_dict = build_dataset(cfg)
    loader_dict = build_loader(cfg, dataset_dict)
    logger.cprint('built dataloader')
    optim_dict = build_optimizer(cfg, model)
    logger.cprint('built optimizer')
    
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()
