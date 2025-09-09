from typing import Any, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
import numpy as np
from einops import rearrange

from hmr4d.model.gvhmr.gvhmr_pl import GvhmrPL
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)


class MotionDiscriminator(nn.Module):
    """Motion Discriminator for distinguishing real vs synthetic motion patterns"""

    def __init__(
        self,
        input_dim=69,  # SMPL pose parameters (72-3 for global rotation)
        hidden_dim=1024,
        num_layers=2,
        output_dim=1,
        feature_pool='attention',
        attention_size=None,
        attention_layers=None,
        attention_dropout=0.1
    ):
        super().__init__()

        self.feature_pool = feature_pool

        # RNN backbone
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Feature pooling
        if feature_pool == 'attention':
            self.attention_size = attention_size or hidden_dim // 4
            self.attention_layers = attention_layers or 1

            attention_modules = []
            for _ in range(self.attention_layers):
                attention_modules.extend([
                    nn.Linear(hidden_dim, self.attention_size),
                    nn.ReLU(),
                    nn.Dropout(attention_dropout)
                ])
            attention_modules.append(nn.Linear(self.attention_size, 1))

            self.attention = nn.Sequential(*attention_modules)

        elif feature_pool == 'max':
            pass  # Will use max pooling
        elif feature_pool == 'avg':
            pass  # Will use average pooling
        else:
            raise ValueError(f"Unknown feature pooling: {feature_pool}")

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, motion_sequence):
        """
        Args:
            motion_sequence: (B, T, input_dim) - pose parameters over time
        Returns:
            confidence: (B, 1) - confidence score [0, 1]
        """
        B, T, _ = motion_sequence.shape

        # RNN encoding
        rnn_out, (h_n, c_n) = self.rnn(motion_sequence)  # (B, T, hidden_dim)

        # Feature pooling
        if self.feature_pool == 'attention':
            # Attention pooling
            attention_weights = self.attention(rnn_out)  # (B, T, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled_features = torch.sum(rnn_out * attention_weights, dim=1)  # (B, hidden_dim)
        elif self.feature_pool == 'max':
            pooled_features = torch.max(rnn_out, dim=1)[0]  # (B, hidden_dim)
        elif self.feature_pool == 'avg':
            pooled_features = torch.mean(rnn_out, dim=1)  # (B, hidden_dim)

        # Classification
        confidence = self.classifier(pooled_features)  # (B, 1)

        return confidence


class DomainAdaptiveGvhmrPL(GvhmrPL):
    """
    Domain Adaptive GVHMR that inherits from GvhmrPL and adds domain adaptation capabilities

    Key features:
    - Alternating synthetic and real data training
    - Motion discriminator for pseudo ground truth generation
    - Confidence-based sample selection
    """

    def __init__(
        self,
        motion_discriminator=None,
        synthetic_weight=1.0,
        real_weight=1.0,
        discriminator_weight=1.0,
        confidence_threshold_start=0.6,
        confidence_threshold_end=1.0,
        confidence_step=0.05,
        num_augmentations=2,
        train_discriminator=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Domain adaptation components
        if motion_discriminator is not None:
            self.motion_discriminator = instantiate(motion_discriminator, _recursive_=False)
        else:
            self.motion_discriminator = MotionDiscriminator()

        # Training weights
        self.synthetic_weight = synthetic_weight
        self.real_weight = real_weight
        self.discriminator_weight = discriminator_weight

        # Confidence threshold scheduling
        self.confidence_threshold_start = confidence_threshold_start
        self.confidence_threshold_end = confidence_threshold_end
        self.confidence_step = confidence_step
        self.current_threshold = confidence_threshold_start

        # Augmentation settings
        self.num_augmentations = num_augmentations
        self.train_discriminator = train_discriminator

        # Iteration counter for threshold scheduling
        self.train_iteration = 0

    def configure_optimizers(self):
        """Configure optimizers for both generator and discriminator"""
        # Get base optimizer from parent
        gen_optimizer_config = super().configure_optimizers()

        # Add discriminator optimizer
        if hasattr(self.motion_discriminator, 'parameters'):
            dis_optimizer = torch.optim.Adam(
                self.motion_discriminator.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )

            if isinstance(gen_optimizer_config, dict):
                return {
                    'generator': gen_optimizer_config,
                    'discriminator': dis_optimizer
                }
            else:
                return {
                    'generator': gen_optimizer_config,
                    'discriminator': dis_optimizer
                }

        return gen_optimizer_config

    def training_step(self, batch, batch_idx):
        """
        Alternating training step between synthetic and real data
        """
        self.train_iteration += 1

        # Update confidence threshold
        self.update_confidence_threshold()

        # Check if this batch contains synthetic or real data
        if 'domain' in batch and batch['domain'] == 'real':
            return self.real_adaptation_step(batch, batch_idx)
        else:
            return self.synthetic_adaptation_step(batch, batch_idx)

    def synthetic_adaptation_step(self, batch, batch_idx):
        """
        Training step for synthetic data (BEDLAM) with full supervision
        """
        # Use parent's training step for synthetic data
        loss_dict = super().training_step(batch, batch_idx)

        # Scale by synthetic weight
        if isinstance(loss_dict, dict) and 'loss' in loss_dict:
            loss_dict['loss'] *= self.synthetic_weight
            loss_dict['synthetic_loss'] = loss_dict['loss']

        Log.info("Synthetic adaptation step completed")
        return loss_dict

    def real_adaptation_step(self, batch, batch_idx):
        """
        Training step for real data (EMDB, 3DPW) with pseudo ground truth
        """
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Generate pseudo ground truth through augmentation and discriminator
        pseudo_gt = self.generate_pseudo_ground_truth(batch)

        # Replace ground truth with pseudo ground truth
        batch_pseudo = batch.copy()
        for key in ['smpl_params_c', 'gt_j3d', 'gt_cr_coco17', 'gt_c_verts437', 'gt_cr_verts437']:
            if key in pseudo_gt:
                batch_pseudo[key] = pseudo_gt[key]

        # Forward pass with pseudo ground truth
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch_pseudo["smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch_pseudo["gt_j3d"] = gt_j3d
            batch_pseudo["gt_cr_coco17"] = gt_j3d - root_
            batch_pseudo["gt_c_verts437"] = gt_verts437
            batch_pseudo["gt_cr_verts437"] = gt_verts437 - root_

        # Compute losses with pseudo ground truth
        pred, _, _ = self.pipeline(batch_pseudo)
        loss_dict = self.compute_losses(pred, batch_pseudo)

        # Motion discriminator loss
        if self.train_discriminator:
            dis_loss = self.compute_discriminator_loss(batch, pred)
            loss_dict.update(dis_loss)

        # Scale by real weight
        total_loss = sum(loss_dict.values()) * self.real_weight
        loss_dict['loss'] = total_loss
        loss_dict['real_loss'] = total_loss

        Log.info("Real adaptation step completed")
        return loss_dict

    def generate_pseudo_ground_truth(self, batch):
        """
        Generate pseudo ground truth using data augmentation and motion discriminator
        """
        pseudo_gts = []

        # Generate multiple augmented versions
        for aug_idx in range(self.num_augmentations):
            # Apply augmentation
            batch_aug = self.apply_augmentation(batch, aug_idx)

            # Forward pass on augmented data
            with torch.no_grad():
                pred, _, _ = self.pipeline(batch_aug)

            # Inverse augmentation to get pseudo GT
            pseudo_gt = self.inverse_augmentation(pred, batch_aug, aug_idx)
            pseudo_gts.append(pseudo_gt)

        # Evaluate with motion discriminator
        if hasattr(self, 'motion_discriminator'):
            confidence_scores = []
            for pgt in pseudo_gts:
                # Extract pose parameters for discriminator
                pose_seq = self.extract_pose_sequence(pgt)
                confidence = self.motion_discriminator(pose_seq)
                confidence_scores.append(confidence)

            # Select high-confidence samples
            return self.select_confident_samples(pseudo_gts, confidence_scores)
        else:
            # Simple averaging if no discriminator
            return self.average_pseudo_gts(pseudo_gts)

    def apply_augmentation(self, batch, aug_idx):
        """Apply data augmentation for pseudo GT generation"""
        # Implement color transform, rigid transform, etc.
        # This is a simplified version - you might want to use the augmentation
        # from the reference code
        batch_aug = batch.copy()

        # Add random rotation and scaling
        if 'features' in batch_aug:
            # Simple augmentation - you can expand this
            noise = torch.randn_like(batch_aug['features']) * 0.01
            batch_aug['features'] = batch_aug['features'] + noise

        return batch_aug

    def inverse_augmentation(self, pred, batch_aug, aug_idx):
        """Inverse augmentation to get pseudo ground truth in original space"""
        # This should inverse the augmentation applied in apply_augmentation
        # For now, return prediction as-is
        return pred

    def extract_pose_sequence(self, pred):
        """Extract pose sequence for motion discriminator"""
        # Extract SMPL pose parameters (excluding global rotation)
        if 'smpl_params_c' in pred and 'body_pose' in pred['smpl_params_c']:
            body_pose = pred['smpl_params_c']['body_pose']  # (B, F, 69)
            return body_pose
        return None

    def select_confident_samples(self, pseudo_gts, confidence_scores):
        """Select high-confidence pseudo ground truth samples"""
        # Convert confidence scores to selection mask
        confidences = torch.stack(confidence_scores, dim=0)  # (num_aug, B, 1)

        # Select samples above threshold
        high_conf_mask = confidences.squeeze(-1) > self.current_threshold  # (num_aug, B)

        # Average high-confidence predictions
        selected_gt = {}
        for key in pseudo_gts[0].keys():
            values = []
            for i, pgt in enumerate(pseudo_gts):
                if high_conf_mask[i].any():
                    values.append(pgt[key])

            if values:
                selected_gt[key] = torch.mean(torch.stack(values), dim=0)
            else:
                # Fallback to first prediction if no high-confidence samples
                selected_gt[key] = pseudo_gts[0][key]

        return selected_gt

    def average_pseudo_gts(self, pseudo_gts):
        """Simple averaging of pseudo ground truths"""
        avg_gt = {}
        for key in pseudo_gts[0].keys():
            values = torch.stack([pgt[key] for pgt in pseudo_gts])
            avg_gt[key] = torch.mean(values, dim=0)
        return avg_gt

    def compute_discriminator_loss(self, batch, pred):
        """Compute motion discriminator loss"""
        if not hasattr(self, 'motion_discriminator'):
            return {}

        # Extract real and fake motion sequences
        real_pose = self.extract_pose_sequence(batch)
        fake_pose = self.extract_pose_sequence(pred)

        if real_pose is None or fake_pose is None:
            return {}

        # Discriminator predictions
        real_conf = self.motion_discriminator(real_pose)
        fake_conf = self.motion_discriminator(fake_pose.detach())

        # Adversarial losses
        real_loss = torch.mean((real_conf - 1.0) ** 2)
        fake_loss = torch.mean(fake_conf ** 2)
        gen_loss = torch.mean((fake_conf - 1.0) ** 2)

        return {
            'discriminator_real_loss': real_loss * self.discriminator_weight,
            'discriminator_fake_loss': fake_loss * self.discriminator_weight,
            'generator_adversarial_loss': gen_loss * self.discriminator_weight
        }

    def update_confidence_threshold(self):
        """Update confidence threshold based on training progress"""
        if self.confidence_step > 0:
            # Calculate progress-based threshold
            max_iterations = self.trainer.max_epochs * self.trainer.num_training_batches
            progress = min(1.0, self.train_iteration / max_iterations)

            self.current_threshold = (
                self.confidence_threshold_start +
                progress * (self.confidence_threshold_end - self.confidence_threshold_start)
            )

    def validation_step(self, batch, batch_idx):
        """Use parent's validation step"""
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Use parent's test step"""
        return super().test_step(batch, batch_idx)
