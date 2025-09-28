import os
import torch
from tqdm import tqdm
from utils import *
from models.utils import PosNegTextEncoder
from models.loss import DenoiserEuclideanLoss
import torch.distributed as dist
import os


class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.logger = running.get("logger", None)
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.sampler = self.loader_dict.get("sampler", None)
        self.optimizer_dict = running["optim_dict"]
        self.optimizer = self.optimizer_dict.get("optimizer", None)
        self.rank = running.get("rank", 0)
        self.world_size = running.get("world_size", 1)
        self.epoch = 0
        
        self.gamma = cfg.training_cfg.get("gamma", 0.9) # gamma for loss functions
        
        # Get device from model
        self.device = next(self.model.parameters()).device
        
        # Initialize text encoder (only on rank 0 for DDP efficiency)
        if self.rank == 0:
            self.posneg_text_encoder = PosNegTextEncoder(device=self.device)
        else:
            self.posneg_text_encoder = None

    def train(self):
        denoiser_euclidean_loss = DenoiserEuclideanLoss()
        self.model.train()
        
        # Only rank 0 logs
        if self.rank == 0 and self.logger:
            self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = self.train_loader
            
        # Handle model attribute access for DDP
        model_module = self.model.module if hasattr(self.model, 'module') else self.model
        if self.epoch > 100 and hasattr(model_module, 'noise_predictor'):
            if hasattr(model_module.noise_predictor, 'scene_encoder'):
                for p in model_module.noise_predictor.scene_encoder.parameters():
                    p.requires_grad = False
        
        # Implement streaming with prefetching
        self.train_epoch_with_streaming(pbar, denoiser_euclidean_loss)
        
        # Only rank 0 saves and logs
        if self.rank == 0 and self.logger:
            self.logger.cprint(f"\nEpoch {self.epoch} completed")
            print("Saving checkpoint\n----------------------------------------\n")
            # Unwrap DDP for saving
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), os.path.join(self.cfg.log_dir, "current_model.t7"))
        
        self.epoch += 1
    
    def train_epoch_with_streaming(self, data_loader, loss_fn):
        """Training loop with streaming/prefetching for optimal performance"""
        # Create iterator for manual batch management
        data_iter = iter(data_loader)
        
        # Prefetch first batch
        try:
            next_batch = next(data_iter)
        except StopIteration:
            return
        
        batch_idx = 0
        total_mse_loss = 0.0
        total_neg_loss = 0.0
        
        while next_batch is not None:
            current_batch = next_batch
            
            # Start loading next batch while processing current (streaming!)
            try:
                next_batch = next(data_iter)
            except StopIteration:
                next_batch = None
            
            # Process current batch
            mse_loss, neg_loss = self.process_batch(current_batch, loss_fn)
            
            total_mse_loss += mse_loss.item() if mse_loss is not None else 0.0
            total_neg_loss += neg_loss.item() if neg_loss is not None else 0.0
            batch_idx += 1
            
            # Update progress bar (only rank 0)
            if self.rank == 0 and hasattr(data_loader, 'set_postfix'):
                data_loader.set_postfix({
                    'MSE': f'{mse_loss.item():.5f}' if mse_loss is not None else 'N/A',
                    'Neg': f'{neg_loss.item():.5f}' if neg_loss is not None else 'N/A'
                })
        
        # Log epoch averages (only rank 0)
        if self.rank == 0 and self.logger and batch_idx > 0:
            avg_mse = total_mse_loss / batch_idx
            avg_neg = total_neg_loss / batch_idx
            self.logger.cprint(f"Epoch {self.epoch} - Avg MSE: {avg_mse:.5f}, Avg Neg: {avg_neg:.5f}")
    
    def process_batch(self, batch, loss_fn):
        """Process a single batch with distributed text processing"""
        _, pc, pos_prompt, neg_prompts, Rt, w = batch
        B = pc.shape[0]
        
        # Move to device
        pc = pc.float().to(self.device, non_blocking=True)
        Rt, w = Rt.float().to(self.device, non_blocking=True), w.float().to(self.device, non_blocking=True)
        noise = torch.randn(B, 7).to(self.device)
        
        # Text processing - only rank 0 processes, then broadcasts
        if self.world_size > 1:
            # Distributed text processing
            if self.rank == 0 and self.posneg_text_encoder is not None:
                with torch.no_grad():
                    pos_prompt_embedding = self.posneg_text_encoder(pos_prompt, type="pos")
                    neg_prompt_embeddings = self.posneg_text_encoder(neg_prompts, type="neg")
                # Broadcast to other ranks
                dist.broadcast(pos_prompt_embedding, src=0)
                dist.broadcast(neg_prompt_embeddings, src=0)
            else:
                # Receive from rank 0
                # Create placeholder tensors with correct shapes
                pos_prompt_embedding = torch.empty((B, 512), device=self.device)  # Adjust size as needed
                neg_prompt_embeddings = torch.empty((B, 4, 512), device=self.device)  # Adjust size as needed
                dist.broadcast(pos_prompt_embedding, src=0)
                dist.broadcast(neg_prompt_embeddings, src=0)
        else:
            # Single GPU - process normally
            if self.posneg_text_encoder is not None:
                with torch.no_grad():
                    pos_prompt_embedding = self.posneg_text_encoder(pos_prompt, type="pos")
                    neg_prompt_embeddings = self.posneg_text_encoder(neg_prompts, type="neg")
            else:
                return None, None  # Skip if no text encoder
        
        # Forward pass
        predicted_noise, neg_prompt_pred, neg_prompt_embeddings_out = self.model(
            Rt, w, pc, pos_prompt_embedding, neg_prompt_embeddings, noise
        )
        
        # Loss computation
        mse_loss, neg_loss = loss_fn(predicted_noise, noise, neg_prompt_pred, neg_prompt_embeddings_out)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss = mse_loss + self.gamma * neg_loss
        total_loss.backward()
        self.optimizer.step()
        
        return mse_loss, neg_loss
            
        self.logger.cprint(f"\nEpoch {self.epoch}, Real-time mse loss: {mse_loss.item():.5f},\
            Real-time neg loss: {neg_loss.item():.5f}")
        print("Saving checkpoint\n----------------------------------------\n")
        torch.save(self.model.state_dict(), os.path.join(self.cfg.log_dir, "current_model.t7"))
        self.epoch += 1
        
    def val(self):
       raise NotImplementedError

    def run(self):
        EPOCH = self.cfg.training_cfg.get('epochs', self.cfg.training_cfg.get('epoch', 100))
        workflow = self.cfg.training_cfg.workflow
        
        while self.epoch < EPOCH:
            # Set epoch for distributed sampler (important for proper shuffling)
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            
            # Training step
            if workflow.get('train', 0) > 0:
                self.train()
            
            # Synchronize all processes at epoch end
            if self.world_size > 1:
                dist.barrier()
            for key, running_epoch in workflow.items():
                epoch_runner = getattr(self, key)
                for _ in range(running_epoch):
                    epoch_runner()
