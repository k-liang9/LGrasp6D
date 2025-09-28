import os
from gorilla.config import Config
from utils import *
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing.spawn import spawn

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    return args    

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size, cfg, samples_map):
    # Setup distributed training
    ddp_setup(rank, world_size)
    
    # Only rank 0 does logging
    if rank == 0:
        logger = IOStream(os.path.join(cfg.log_dir, "run.log"))
        if cfg.get("seed") is not None:
            set_random_seed(cfg.seed)
            logger.cprint("Set seed to %d" % cfg.seed)
        logger.cprint("Training from scratch with DDP!")
    else:
        logger = None
    
    # Build model and move to correct GPU
    model = build_model(cfg)
    model = model.to(f'cuda:{rank}')
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Build dataset and distributed loader
    train_dataset = build_dataset(cfg, samples_map)
    
    if rank == 0 and logger and train_dataset is not None:
        logger.cprint(f"Dataset built with {len(train_dataset)} samples")
    
    loader_dict = build_loader(cfg, {'train_set': train_dataset}, world_size=world_size, rank=rank)
    train_loader = loader_dict['train_loader']
    
    if rank == 0 and logger:
        logger.cprint('Built distributed dataloader')
    
    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    
    if rank == 0 and logger:
        logger.cprint('Built optimizer')
    
    # Training setup
    training = dict(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        logger=logger,
        rank=rank,
        world_size=world_size
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()
    
    ddp_cleanup()
    
    

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # Build sample mappings once (expensive operation)  
    samples_map = build_dataset_sample_mappings(cfg, is_train=True)
    
    # Setup for distributed or single GPU training
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Multi-GPU distributed training
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Built samples map with {len(samples_map)} samples")
        spawn(train_process, args=(world_size, cfg, samples_map), nprocs=world_size, join=True)
    else:
        # Single GPU training
        print("Starting single GPU training")
        train_process(0, 1, cfg, samples_map)