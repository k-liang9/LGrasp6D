from dataset import Grasp6DDataset_Train, Grasp6DDataset
from models import *
from utils.config_utils import simple_weights_init
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam


model_pool = {
    "Denoiser": Denoiser
}

optimizer_pool = {
    "adam": Adam
}

init_pool = {
    "simple_weights_init": simple_weights_init
}

def build_dataset_sample_mappings(cfg, is_train):
    log_dir = cfg.log_dir
    dataset_path = cfg.data.dataset_path
    num_workers = cfg.hardware.num_cpus
    from dataset.Grasp6DDataset import Grasp6DDataset
    samples_map = Grasp6DDataset.create_sample_mappings(dataset_path, log_dir, is_train, num_workers)
    return samples_map

def build_dataset(cfg, samples_map, is_train=True):
    dataset_path = cfg.data.dataset_path
    log_dir = cfg.data.log_dir
    num_workers = cfg.train.num_workers
    
    # Use pre-computed samples_map (passed as parameter)
    # Create the dataset with correct parameters
    num_neg_prompts = getattr(cfg.data, 'num_neg_prompts', 4)
    
    train_set = Grasp6DDataset_Train(
        dataset_path=dataset_path,
        log_dir=log_dir, 
        samples_mapping=samples_map,
        num_neg_prompts=num_neg_prompts
    )
    
    return train_set

def build_loader(cfg, dataset_dict, rank=0, world_size=1):
    """
    Function to build the loader with distributed sampling and streaming optimizations
    """
    train_set = dataset_dict["train_set"]
    
    # Create distributed sampler if using multiple GPUs
    sampler = None
    shuffle = True
    
    if world_size > 1:
        sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        shuffle = False  # DistributedSampler handles shuffling
    
    # Optimize workers and add streaming features
    num_workers = max(1, cfg.hardware.get('num_cpus', 4) // world_size)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.training_cfg.batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False, 
        num_workers=num_workers,
        prefetch_factor=2,      # Prefetch 2 batches per worker
        pin_memory=True,        # Faster GPU transfer
        persistent_workers=True # Keep workers alive between epochs
    )
    
    loader_dict = dict(
        train_loader=train_loader,
        sampler=sampler
    )
    
    return loader_dict


def build_model(cfg):
    """
    Function to build the model.
    """
    if hasattr(cfg, "model"):
        model_info = cfg.model
        weights_init = model_info.get("weights_init", None)
        model_name = model_info.type
        model_cls = model_pool[model_name]
        
        if model_name in ["Denoiser"]:
            betas = model_info.get("betas")
            n_T = model_info.get("n_T")
            drop_prob = model_info.get("drop_prob") 
            model = model_cls(n_T, betas, drop_prob)
        else:
            raise ValueError("Name of model does not exist!")
        if weights_init is not None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_optimizer(cfg, model):
    """
    Function to build the optimizer.
    """
    if hasattr(cfg, "optimizer"):
        optimizer_info = cfg.optimizer
        optimizer_type = optimizer_info.type
        optimizer_info.pop("type")
        optimizer_cls = optimizer_pool[optimizer_type]
        optimizer = optimizer_cls(model.parameters(), **optimizer_info)
        optim_dict = dict(
            optimizer=optimizer
        )
        return optim_dict
    else:
        raise ValueError("Configuration does not have optimizer config!")
