"""
Training Module - Loss functions, training loops, and optimization.
"""

import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR

logger = logging.getLogger(__name__)


class HybridLoss(nn.Module):
    """
    Hybrid loss combining classification and regression.
    """
    
    def __init__(
        self,
        classification_weight: float = 0.7,
        regression_weight: float = 0.3,
        label_smoothing: float = 0.1
    ):
        """
        Initialize hybrid loss.
        
        Args:
            classification_weight: Weight for BCE loss
            regression_weight: Weight for MSE loss
            label_smoothing: Label smoothing for BCE
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        target_direction: torch.Tensor,
        target_return: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss.
        
        Args:
            logits: Model output (batch_size, 1)
            target_direction: Binary targets 0 or 1 (batch_size, 1)
            target_return: Actual returns (batch_size, 1) or None
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Apply label smoothing
        target_direction_smooth = target_direction * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Classification loss
        class_loss = self.bce_loss(logits, target_direction_smooth)
        
        # Regression loss (if target returns provided)
        if target_return is not None:
            reg_loss = self.mse_loss(logits, target_return)
            total_loss = (
                self.classification_weight * class_loss +
                self.regression_weight * reg_loss
            )
        else:
            total_loss = class_loss
            reg_loss = torch.tensor(0.0)
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': class_loss.item(),
            'regression': reg_loss.item() if target_return is not None else 0.0
        }
        
        return total_loss, loss_dict


class Trainer:
    """
    Training class for the hybrid temporal model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: torch device
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        train_config = config.get('training', {})
        
        # Loss function
        loss_config = train_config.get('loss', {})
        self.criterion = HybridLoss(
            classification_weight=loss_config.get('loss_weights', {}).get('classification', 0.7),
            regression_weight=loss_config.get('loss_weights', {}).get('regression', 0.3),
            label_smoothing=train_config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        lr = train_config.get('learning_rate', 1e-3)
        l2_decay = train_config.get('regularization', {}).get('l2_weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=l2_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=l2_decay)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=l2_decay)
        
        # Learning rate scheduler
        schedule_type = train_config.get('learning_rate_schedule', 'cosine')
        num_epochs = train_config.get('num_epochs', 100)
        
        if schedule_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif schedule_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        elif schedule_type == 'linear':
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0, total_iters=num_epochs)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Gradient clipping
        self.gradient_clip = train_config.get('gradient_clipping', 1.0)
        
        # Regularization
        self.dropout_rate = train_config.get('regularization', {}).get('dropout_rate', 0.3)
        
        logger.info(f"Trainer initialized: optimizer={optimizer_name}, lr={lr}, schedule={schedule_type}")
    
    def train_epoch(
        self,
        data_loader
    ) -> Dict[str, float]:
        """
        Train one epoch.
        
        Args:
            data_loader: DataLoader with batches
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_regression_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Unpack batch
            historical = batch['historical'].to(self.device)
            sentiment = batch['sentiment'].to(self.device)
            gnn_embeddings = batch['gnn_embeddings'].to(self.device)
            target_direction = batch['direction'].to(self.device)
            target_return = batch.get('return')
            
            if target_return is not None:
                target_return = target_return.to(self.device)
            
            # Forward pass
            output = self.model(historical, sentiment, gnn_embeddings)
            
            # Loss computation
            loss, loss_dict = self.criterion(output, target_direction, target_return)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total']
            total_classification_loss += loss_dict['classification']
            total_regression_loss += loss_dict['regression']
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(data_loader)}: loss={loss.item():.4f}")
        
        # Step scheduler
        self.scheduler.step()
        
        avg_metrics = {
            'total_loss': total_loss / max(num_batches, 1),
            'classification_loss': total_classification_loss / max(num_batches, 1),
            'regression_loss': total_regression_loss / max(num_batches, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return avg_metrics
    
    def validate(
        self,
        data_loader
    ) -> Dict[str, float]:
        """
        Validation pass.
        
        Args:
            data_loader: Validation DataLoader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch
                historical = batch['historical'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                gnn_embeddings = batch['gnn_embeddings'].to(self.device)
                target_direction = batch['direction'].to(self.device)
                target_return = batch.get('return')
                
                if target_return is not None:
                    target_return = target_return.to(self.device)
                
                # Forward pass
                output = self.model(historical, sentiment, gnn_embeddings)
                
                # Loss
                loss, loss_dict = self.criterion(output, target_direction, target_return)
                total_loss += loss_dict['total']
                
                # Accuracy
                predictions = (output > 0.5).float()
                total_correct += (predictions == target_direction).sum().item()
                total_samples += target_direction.shape[0]
        
        accuracy = total_correct / max(total_samples, 1)
        
        metrics = {
            'loss': total_loss / max(len(data_loader), 1),
            'accuracy': accuracy
        }
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_freq: int = 5
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_freq: Frequency to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train loss: {train_metrics['total_loss']:.4f}")
            history['train_loss'].append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val loss: {val_metrics['loss']:.4f}, Val accuracy: {val_metrics['accuracy']:.4f}")
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Early stopping
            if val_metrics['loss'] < history['best_val_loss']:
                history['best_val_loss'] = val_metrics['loss']
                history['best_epoch'] = epoch
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        logger.info(f"Loaded checkpoint from {path}")
