
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from utils.overcomplicated_torch import OvercomplicatedDataset
from utils.sequence_dataloader import OvercomplicatedDataset
from utils.sequence_dataloader import collate_fn
from glob import glob
from tqdm import tqdm
import numpy as np
import timm
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def unfreeze_layers_gradually(model, current_epoch, total_unfreeze_epochs=8, start_epoch=4):
    """Gradually unfreeze backbone layers over several epochs"""
    logging.debug(f"Unfreezing layers at epoch {current_epoch} with start_epoch={start_epoch}, total_unfreeze_epochs={total_unfreeze_epochs}")
    if current_epoch < start_epoch:
        # Keep all backbone frozen
        logging.debug(f"Epoch {current_epoch}: Backbone layers frozen")
        for param in model.backbone.parameters():
            param.requires_grad = False
        if current_epoch == start_epoch - 1 or current_epoch == 0:
            logging.info(f"Epoch {current_epoch}: Backbone layers frozen")
        return False
    
    if current_epoch >= start_epoch + total_unfreeze_epochs:
        # Unfreeze all backbone
        logging.debug(f"Epoch {current_epoch}: All backbone layers unfrozen")
        for param in model.backbone.parameters():
            param.requires_grad = True
        if current_epoch - 1 < start_epoch + total_unfreeze_epochs:
            logging.info(f"Epoch {current_epoch}: All backbone layers unfrozen")
        return False

    # Calculate how many layers to unfreeze
    progress = (current_epoch - start_epoch) / total_unfreeze_epochs
    
    # Get list of backbone modules in reverse order (unfreeze from last to first)
    backbone_modules = list(model.backbone.named_children())
    num_modules = len(backbone_modules)
    num_modules = 0
    def count_modules(module):
        nonlocal num_modules
        if isinstance(module, nn.Module):
            # check if the module is a normalization layer and do not count it
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                return
            for child in module.children():
                count_modules(child)
            if len(list(module.children())) == 0:
                # If this module has no children, count it as a module
                num_modules += 1

    count_modules(model.backbone)

    logging.debug(f"Epoch {current_epoch}: Found {num_modules} backbone modules")

    modules_to_unfreeze = int(progress * num_modules)
    modules_to_keep_frozen = num_modules - modules_to_unfreeze

    logging.debug(f"Epoch {current_epoch}: Unfreezing {modules_to_unfreeze}/{num_modules} backbone modules (progress: {progress:.2f})")
    
    count = 0
    start_unfreezing = False

    def unfreeze_module(module):
        """Recursively unfreeze a module and its children"""
        # generators can't be reversed, so we need to iterate through all the layers using modules_to_keep_frozen, and then just unfreeze everything after that
        nonlocal count, start_unfreezing
        if isinstance(module, nn.Module):
            # check if the module is a normalization layer and do not unfreeze it
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                logging.debug(f"Skipping normalization layer")
                return
            for child in module.children():
                logging.debug(f"Unfreezing child module")
                unfreeze_module(child)
            if len(list(module.children())) > 0:
                logging.debug(f"Module has children, skipping unfreeze")
                return
            count += 1
            if count > modules_to_keep_frozen and not start_unfreezing:
                # If we have reached the number of modules to unfreeze, start unfreezing
                start_unfreezing = True
                logging.debug(f"Starting to unfreeze modules after {modules_to_keep_frozen} modules")
            if start_unfreezing: # only unfreeze if we are past the modules_to_keep_frozen count and the module is does not have children or is not a normalization layer
                logging.debug(f"Unfreezing module: (count: {count}/{num_modules})")
                for param in module.parameters():
                    param.requires_grad = True
                logging.debug(f"Unfroze module: (count: {count}/{num_modules})")
    
    unfreeze_module(model.backbone)

    logging.debug(f"Epoch {current_epoch}: Unfroze {count} backbone modules")
                

    if modules_to_unfreeze > 0 and modules_to_unfreeze > int((current_epoch - start_epoch - 1) / total_unfreeze_epochs * num_modules):
        # Log the number of modules unfrozen if there was a change in the number of unfrozen modules
        if modules_to_unfreeze > 0:
            logging.info(f"Epoch {current_epoch}: Unfrozen {modules_to_unfreeze}/{num_modules} backbone modules")
            return True
        else:
            logging.info(f"Epoch {current_epoch}: No backbone modules unfrozen")
            return False
    elif modules_to_unfreeze == 0 and count == num_modules:
        # If we have already unfrozen all modules, log that there was no change
        logging.debug(f"Epoch {current_epoch}: All backbone modules already unfrozen")
        return False
    else:
        logging.debug(f"Epoch {current_epoch}: No change in backbone modules unfrozen")
        return False



def main():
    # Argument parsing
    cwd = os.getcwd()
    def_path = lambda p: os.path.join(cwd, p)
    parser = argparse.ArgumentParser(description='Train a model (PyTorch version).')
    parser.add_argument('--image_frames', type=int, default=60, help='number of frames per video')
    parser.add_argument('--image_size', type=int, default=236, help='size of image') # 384 is image size. 236 is the convnext_tiny default size
    parser.add_argument('--label_data_root', type=str, default=def_path('dataset/universal_labels'), help='path to label data')
    parser.add_argument('--image_data_root', type=str, default=def_path('dataset/features512'), help='path to image data')
    parser.add_argument('--validation_image_data_root', type=str, default=def_path('dataset/features512_validation'), help='path to validation image data')
    parser.add_argument('--validation_new_label_data_root', type=str, default=def_path('dataset/universal_labels_validation'), help='path to validation label data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, use 16 per 24gb of vram')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--existing_model', type=str, default=None, help='path to existing model to continue training')    
    parser.add_argument('--train_backbone', action='store_true', help='whether to train the backbone model')
    parser.add_argument('--dataset_folder', type=str, default=def_path('dataset'), help='path to dataset folder')
    parser.add_argument('--validation_dataset_folder', type=str, default=def_path('dataset/validation'), help='path to validation dataset folder')
    
    # Learning rate and early stopping arguments
    parser.add_argument('--lr_patience', type=int, default=50, help='patience for learning rate reduction')
    parser.add_argument('--lr_factor', type=float, default=0.99, help='factor to reduce learning rate by')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='patience for early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001, help='minimum change for early stopping')
    parser.add_argument('--disable_early_stopping', action='store_true', help='disable early stopping')
    
    args = parser.parse_args()
    
    unfreeze_layers_gradually_from_epoch = 0
    unfreeze_layers_gradually_over_epochs = 0
    merge_with_best_every_epochs = 20
    checkpoint_every_steps = 1000

    # Datasets and loaders
    # train_dataset = OvercomplicatedDataset(args.label_data_root, args.image_data_root, duration=args.image_frames, croppct=0.2, augment=True, image_size=args.image_size, max_sequences=15000)
    # val_dataset = OvercomplicatedDataset(args.validation_new_label_data_root, args.validation_image_data_root, duration=args.image_frames, croppct=0, augment=False, image_size=args.image_size, max_sequences=1500)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)


    train_dataset = OvercomplicatedDataset(
        dataset_folder=args.dataset_folder,
        duration=args.image_frames,
        croppct=0.2,
        augment=False,
        image_size=args.image_size,
        max_sequences=25000,
        random_seed=42
    )
    val_dataset = OvercomplicatedDataset(
        dataset_folder=args.validation_dataset_folder,
        duration=args.image_frames,
        croppct=0,
        augment=False,
        image_size=args.image_size,
        max_sequences=2500,
        random_seed=24
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modern Video Transformer Architecture
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:x.size(0), :]

    class VideoTransformerRegressor(nn.Module):
        def __init__(self, image_frames, image_size):
            super().__init__()
            self.image_frames = image_frames
            self.image_size = image_size
            
            # Spatial feature extractor
            self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
            self.backbone.eval()
            if not args.train_backbone:
                # Freeze backbone parameters if not training
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True
            
            # Feature dimensions
            self.feature_dim = 768  # ConvNeXt tiny
            self.d_model = 256  # Transformer dimension
            
            # Spatial feature processing
            self.feature_proj = nn.Sequential(
                nn.Linear(self.feature_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.Dropout(0.1)
            )
            
            # Temporal modeling with Transformer
            self.pos_encoding = PositionalEncoding(self.d_model, max_len=image_frames)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            
            # Multi-scale temporal attention
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Regression head with residual connections
            self.regression_head = nn.Sequential(
                nn.Linear(self.d_model, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, image_frames)
            )
            
            # Global context for sequence-level features
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.sequence_proj = nn.Linear(self.d_model, image_frames)
            
        def forward(self, x):
            B, T, C, H, W = x.shape
            
            # Extract spatial features
            x = x.view(B * T, C, H, W)
            with torch.no_grad():
                spatial_feats = self.backbone.forward_features(x)
                spatial_feats = spatial_feats.mean([2, 3])  # Global average pooling
            
            # Project to transformer dimension
            spatial_feats = self.feature_proj(spatial_feats)  # (B*T, d_model)
            spatial_feats = spatial_feats.view(B, T, -1)  # (B, T, d_model)
            
            # Add positional encoding
            spatial_feats = spatial_feats.transpose(0, 1)  # (T, B, d_model)
            spatial_feats = self.pos_encoding(spatial_feats)
            spatial_feats = spatial_feats.transpose(0, 1)  # (B, T, d_model)
            
            # Transformer encoding for temporal dependencies
            temporal_feats = self.transformer_encoder(spatial_feats)  # (B, T, d_model)
            
            # Additional temporal attention
            attended_feats, _ = self.temporal_attention(
                temporal_feats, temporal_feats, temporal_feats
            )
            
            # Residual connection
            temporal_feats = temporal_feats + attended_feats
            
            # Global sequence context
            global_context = self.global_pool(temporal_feats.transpose(1, 2)).squeeze(-1)  # (B, d_model)
            global_pred = self.sequence_proj(global_context)  # (B, image_frames)
            
            # Frame-wise predictions
            frame_preds = self.regression_head(temporal_feats.mean(dim=1))  # (B, image_frames)
            
            # Combine global and local predictions
            output = 0.7 * frame_preds + 0.3 * global_pred
            output = torch.relu(output)  # Ensure non-negative (0-1 range for jump apex)
            
            return output
        

    def merge_models(model1, model2, alpha=0.5):
        """Merge two models by averaging their parameters"""
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1.data.copy_(alpha * param1.data + (1 - alpha) * param2.data)
        return model1

    model = VideoTransformerRegressor(args.image_frames, args.image_size).to(device)
    if args.existing_model:
        model.load_state_dict(torch.load(args.existing_model, map_location=device))
        if not args.train_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
        model.to(device)

    print("Testing merging models...")
    # # Test merging models
    # if args.existing_model:
    #     best_model_path = f'models/best_model_{args.image_frames}_{args.image_size}.pth'
    #     if os.path.exists(best_model_path):
    #         best_model = VideoTransformerRegressor(args.image_frames, args.image_size).to(device)
    #         best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    #         model = merge_models(model, best_model, alpha=0.5)
    #         print("Merged with best model for initial training")
    #         del best_model
    #     else:
    #         print("No best model found for merging, starting from scratch")

    # Modern optimizer and training setup
    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler - reduce on plateau like TensorFlow version
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.lr_min,
        threshold=args.early_stopping_min_delta,
    )
    
    
    # TensorBoard logging
    log_dir = f"logs/{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    # Create checkpoints directory
    checkpoint_dir = f"checkpoints/{int(time.time())}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    steps_per_epoch = len(train_loader)
    current_step = 0
    unfreeze_layers_gradually_from_step = unfreeze_layers_gradually_from_epoch * steps_per_epoch
    unfreeze_layers_gradually_over_steps = unfreeze_layers_gradually_over_epochs * steps_per_epoch  # Unfreeze layers gradually
    last_best_epoch = 0 * steps_per_epoch
    last_merged_epoch = 0 * steps_per_epoch
    merge_with_best_every = merge_with_best_every_epochs * steps_per_epoch
    merge_with_best = True  # Flag to control merging with best model



    # Smooth L1 loss is often better for regression than MSE
    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if torch.cuda.is_available() else None    # Training loop
    best_val_loss = float('inf')

    model.eval()
    val_loss = 0
    val_batches = 0
    epoch = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Validation'):
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if scaler else torch.float32):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_batches += 1
    val_loss /= val_batches
    best_val_loss = val_loss
    print(f'Initial validation loss: {val_loss:.6f}')
    last_best_epoch = 0  # Reset last best epoch since we start from scratch
    print("Starting training...")


    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        model.train()
        logging.info(f"Training model on device: {device}")
        train_loss = 0
        train_batches = 0

        logging.debug(f"Current step: {current_step}, epoch: {epoch}, steps per epoch: {steps_per_epoch}")
        logging.debug(f"Merge with best every: {merge_with_best_every}, last best epoch: {last_best_epoch}, last merged epoch: {last_merged_epoch}, current epoch: {epoch * steps_per_epoch}")
        logging.debug(f"Unfreeze layers gradually from step: {unfreeze_layers_gradually_from_step}, current step: {current_step}")
        



        if epoch * steps_per_epoch >= unfreeze_layers_gradually_from_step:
            pass
        else:
            if os.path.exists('start_unfreezing.txt'): # start unfreezing early
                unfreeze_layers_gradually_from_step = epoch * steps_per_epoch
                try:
                    with open('start_unfreezing.txt', 'w') as f:
                        unfreeze_text = f.read()
                        # check if the text is an integer
                        if unfreeze_text.isdigit(): # if an integer has been written to the file, use it
                            unfreeze_layers_gradually_over_steps = int(unfreeze_text) * steps_per_epoch
                    
                except Exception as e:
                    pass
                try:
                    os.remove('start_unfreezing.txt')
                except Exception as e:
                    pass
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Training')):
            layers_unfrozen = False
            current_step += 1
            
            if current_step >= unfreeze_layers_gradually_from_step and args.train_backbone and current_step < unfreeze_layers_gradually_from_step + unfreeze_layers_gradually_over_steps * steps_per_epoch:
                layers_unfrozen = unfreeze_layers_gradually(model, current_step, total_unfreeze_epochs=unfreeze_layers_gradually_over_steps, start_epoch=unfreeze_layers_gradually_from_step)
            if layers_unfrozen:
                # increase the learning rate if we just unfroze layers
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.1, 1e-5)
            
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if scaler else torch.float32):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # TensorBoard logging - per batch like TensorFlow version
            writer.add_scalar('Loss/Train_Batch', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)

            if current_step % checkpoint_every_steps == 0:
                # Save checkpoint every checkpoint_every_steps steps
                checkpoint_path = os.path.join(checkpoint_dir, f'last_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss / train_batches,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                logging.info(f"Checkpoint saved at step {current_step} to {checkpoint_path}")
        
        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Validation'):
                frames, labels = batch
                frames = frames.to(device)
                labels = labels.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if scaler else torch.float32):
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= val_batches
        
        # TensorBoard logging - per epoch
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
        writer.add_scalar('Learning_Rate_Epoch', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Step the learning rate scheduler with validation loss
        scheduler.step(val_loss)
        

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/best_model_{args.image_frames}_{args.image_size}.pth')
            print(f"New best model saved with validation loss: {val_loss:.6f}")
            last_best_epoch = epoch * steps_per_epoch

        
        # Save checkpoint every epoch like TensorFlow version
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        
        # Also save last model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/last_model_{args.image_frames}_{args.image_size}.pth')
      
        if val_loss > best_val_loss:
            if merge_with_best and (epoch * steps_per_epoch - last_best_epoch) >= merge_with_best_every and (epoch * steps_per_epoch) > (unfreeze_layers_gradually_from_step + unfreeze_layers_gradually_over_steps + merge_with_best_every) and (epoch * steps_per_epoch) > last_merged_epoch + merge_with_best_every:
                last_merged_epoch = epoch * steps_per_epoch
                best_model_path = f'models/best_model_{args.image_frames}_{args.image_size}.pth'
                if os.path.exists(best_model_path):
                    print(f"Merging with best model from epoch {last_best_epoch // steps_per_epoch + 1}")
                    best_model = VideoTransformerRegressor(args.image_frames, args.image_size).to(device)
                    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
                    model = merge_models(model, best_model, alpha=0.9) # Merge with 90% current model, 10% best model
                    print(f"Merged with best model at epoch {epoch+1}")
                    del best_model
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
