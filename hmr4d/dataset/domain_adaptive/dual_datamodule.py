import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
import torch
import itertools
import random
from typing import Iterator, Optional


class DualDataLoader:
    """
    DataLoader that alternates between synthetic and real datasets
    """

    def __init__(self, synthetic_loader, real_loader, alternating_ratio=1):
        """
        Args:
            synthetic_loader: DataLoader for synthetic data (BEDLAM)
            real_loader: DataLoader for real data (EMDB, 3DPW)
            alternating_ratio: How many real batches per synthetic batch
        """
        self.synthetic_loader = synthetic_loader
        self.real_loader = real_loader
        self.alternating_ratio = alternating_ratio

        # Create iterators
        self.synthetic_iter = iter(synthetic_loader)
        self.real_iter = iter(real_loader)

        # Track current step for alternation
        self.current_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Return next batch, alternating between synthetic and real"""
        self.current_step += 1

        # Determine whether to return synthetic or real data
        if self.current_step % (self.alternating_ratio + 1) == 0:
            # Return synthetic data
            try:
                batch = next(self.synthetic_iter)
                batch['domain'] = 'synthetic'
                return batch
            except StopIteration:
                # Reset synthetic iterator
                self.synthetic_iter = iter(self.synthetic_loader)
                batch = next(self.synthetic_iter)
                batch['domain'] = 'synthetic'
                return batch
        else:
            # Return real data
            try:
                batch = next(self.real_iter)
                batch['domain'] = 'real'
                return batch
            except StopIteration:
                # Reset real iterator
                self.real_iter = iter(self.real_loader)
                batch = next(self.real_iter)
                batch['domain'] = 'real'
                return batch

    def __len__(self):
        """Return approximate length"""
        return max(len(self.synthetic_loader), len(self.real_loader))


class DomainAdaptiveDataModule(pl.LightningDataModule):
    """
    DataModule for domain adaptive training with separate synthetic and real datasets
    """

    def __init__(
        self,
        synthetic_data_config,
        real_data_config,
        alternating_ratio=1,
        val_data_config=None,
        test_data_config=None
    ):
        """
        Args:
            synthetic_data_config: Config for synthetic dataset (BEDLAM)
            real_data_config: Config for real datasets (EMDB, 3DPW)
            alternating_ratio: How many real batches per synthetic batch
            val_data_config: Validation dataset config
            test_data_config: Test dataset config
        """
        super().__init__()

        self.synthetic_data_config = synthetic_data_config
        self.real_data_config = real_data_config
        self.alternating_ratio = alternating_ratio
        self.val_data_config = val_data_config
        self.test_data_config = test_data_config

        # Dataset instances (will be created in setup)
        self.synthetic_dataset = None
        self.real_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""

        if stage == "fit" or stage is None:
            # Setup training datasets
            self.synthetic_dataset = instantiate(
                self.synthetic_data_config,
                _recursive_=False
            )

            # Handle multiple real datasets
            if isinstance(self.real_data_config, list):
                # Multiple real datasets - create combined dataset
                real_datasets = []
                for config in self.real_data_config:
                    dataset = instantiate(config, _recursive_=False)
                    real_datasets.append(dataset)
                self.real_dataset = CombinedDataset(real_datasets)
            else:
                # Single real dataset
                self.real_dataset = instantiate(
                    self.real_data_config,
                    _recursive_=False
                )

            # Setup validation dataset
            if self.val_data_config is not None:
                self.val_dataset = instantiate(
                    self.val_data_config,
                    _recursive_=False
                )

        if stage == "test" or stage is None:
            # Setup test dataset
            if self.test_data_config is not None:
                self.test_dataset = instantiate(
                    self.test_data_config,
                    _recursive_=False
                )

    def train_dataloader(self):
        """Return dual dataloader for training"""

        synthetic_loader = DataLoader(
            self.synthetic_dataset,
            batch_size=getattr(self.synthetic_data_config, 'batch_size', 32),
            shuffle=True,
            num_workers=getattr(self.synthetic_data_config, 'num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        real_loader = DataLoader(
            self.real_dataset,
            batch_size=getattr(self.real_data_config, 'batch_size', 32),
            shuffle=True,
            num_workers=getattr(self.real_data_config, 'num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        return DualDataLoader(
            synthetic_loader,
            real_loader,
            self.alternating_ratio
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=getattr(self.val_data_config, 'batch_size', 32),
            shuffle=False,
            num_workers=getattr(self.val_data_config, 'num_workers', 4),
            pin_memory=True
        )

    def test_dataloader(self):
        """Return test dataloader"""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=getattr(self.test_data_config, 'batch_size', 32),
            shuffle=False,
            num_workers=getattr(self.test_data_config, 'num_workers', 4),
            pin_memory=True
        )


class CombinedDataset(Dataset):
    """
    Combine multiple datasets into one
    """

    def __init__(self, datasets):
        """
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)

        # Create cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for length in self.lengths:
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + length
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """Get item from appropriate dataset"""
        if idx < 0:
            idx += self.total_length

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                dataset_idx = i
                break

        # Calculate local index within the dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        # Get item from appropriate dataset
        return self.datasets[dataset_idx][local_idx]


class MotionDiscriminatorDataset(Dataset):
    """
    Dataset specifically for training motion discriminator
    Provides real motion sequences from multiple datasets
    """

    def __init__(self, motion_datasets, sequence_length=16):
        """
        Args:
            motion_datasets: List of datasets containing motion data
            sequence_length: Length of motion sequences
        """
        self.datasets = motion_datasets
        self.sequence_length = sequence_length

        # Extract motion sequences from all datasets
        self.motion_sequences = []
        self._extract_motion_sequences()

    def _extract_motion_sequences(self):
        """Extract motion sequences from datasets"""
        for dataset in self.datasets:
            # Iterate through dataset and extract motion sequences
            for i in range(len(dataset)):
                try:
                    item = dataset[i]

                    # Extract pose parameters
                    if 'smpl_params_c' in item and 'body_pose' in item['smpl_params_c']:
                        pose_seq = item['smpl_params_c']['body_pose']  # (T, 69)

                        if pose_seq.shape[0] >= self.sequence_length:
                            # Split into overlapping sequences
                            for start_idx in range(0, pose_seq.shape[0] - self.sequence_length + 1,
                                                 self.sequence_length // 2):
                                seq = pose_seq[start_idx:start_idx + self.sequence_length]
                                self.motion_sequences.append(seq)

                except Exception as e:
                    # Skip problematic items
                    continue

    def __len__(self):
        return len(self.motion_sequences)

    def __getitem__(self, idx):
        """Return motion sequence"""
        return {
            'theta': self.motion_sequences[idx],  # (T, 69)
            'domain': 'real'
        }
