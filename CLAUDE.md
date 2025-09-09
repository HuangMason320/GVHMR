# CLAUDE.md

## tools for claude code
- 你可以使用 「gemini -p "xxx"」來呼叫 gemini cli 這個工具做事情， gemini cli 的上下文 token 很大，你可以用它找專案裡的程式碼，上網查資料等。但禁止使用它修改或刪除檔案。以下是一個使用範例
- Bash(gemini -p "找出專案裡使用 xAI 的地方")

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GVHMR (Gravity-View Human Motion Recovery) is a PyTorch-based research project for world-grounded human motion recovery using gravity-view coordinates. The project implements a transformer-based architecture for 3D human motion estimation from monocular video input.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

### Training and Testing
```bash
# Train model
python tools/train.py exp=gvhmr/mixed/mixed

# Test on individual datasets
python tools/train.py global/task=gvhmr/test_3dpw exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
python tools/train.py global/task=gvhmr/test_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
python tools/train.py global/task=gvhmr/test_emdb exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# Test all datasets at once
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

### Demo Inference
```bash
# Single video demo (use -s to skip visual odometry for static camera)
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s

# Batch demo on folder
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Code Quality
```bash
# Format code with Black
black hmr4d/ tools/
```

## Project Architecture

### Core Components

**Training Pipeline (`tools/train.py`)**
- Uses PyTorch Lightning framework with Hydra configuration management
- Entry point for both training and testing workflows
- Handles model initialization, data loading, and callback management

**GVHMR Model (`hmr4d/model/gvhmr/`)**
- `gvhmr_pl.py`: PyTorch Lightning module wrapping the core pipeline
- `pipeline/gvhmr_pipeline.py`: Main inference pipeline implementing the GVHMR algorithm
- `gvhmr_pl_demo.py`: Streamlined demo version for inference

**Network Architecture (`hmr4d/network/`)**
- `gvhmr/relative_transformer.py`: Core transformer architecture with RoPE (Rotary Position Embedding)
- `hmr2/`: HMR2.0 components for human mesh recovery
- `base_arch/`: Shared architectural components (transformers, embeddings)

**Dataset Handling (`hmr4d/dataset/`)**
- Supports multiple datasets: BEDLAM, EMDB, H36M, RICH, 3DPW
- Each dataset has its own module with preprocessing utilities
- `imgfeat_motion/` and `pure_motion/` for different input modalities

**Preprocessing (`hmr4d/utils/preproc/`)**
- `vitpose.py`: 2D pose estimation using ViTPose
- `slam.py`: Visual odometry and camera tracking
- `simple_vo.py`: Simplified visual odometry (default, more efficient than DPVO)
- `tracker.py`: Object tracking utilities

### Configuration System

Uses Hydra for hierarchical configuration:
- `hmr4d/configs/`: Main configuration directory
- `exp/`: Experiment configurations combining model, data, and training settings
- `global/task/`: Task-specific overrides for testing different datasets
- `data/mocap/`: Data loading configurations

### Key Utilities

**Body Models (`hmr4d/utils/body_model/`)**
- SMPL/SMPLX body model implementations
- Joint regressor utilities for different keypoint formats

**Geometry (`hmr4d/utils/geo/`)**
- Camera projection and transformation utilities
- Augmentation and pose processing functions

**Visualization (`hmr4d/utils/vis/`)**
- Rendering utilities for 3D mesh visualization
- Integration with wis3d for web-based visualization

## Important Notes

- The project requires CUDA-enabled PyTorch for GPU acceleration
- Type checking is disabled in pyrightconfig.json
- Uses mixed precision training with PyTorch Lightning
- SimpleVO is now the default visual odometry method (more efficient than DPVO)
- Model checkpoints and datasets must be downloaded separately from Google Drive
- SMPL/SMPLX body models require separate registration and download
