# Domain Adaptive GVHMR Training

本文件說明如何使用新的 domain adaptation 功能來訓練 GVHMR 模型。

## 概述

Domain Adaptive GVHMR 實現了基於參考論文的訓練架構，主要特點：

1. **交替訓練**: 在 synthetic data (BEDLAM) 和 real data (EMDB, 3DPW) 之間交替
2. **Motion Discriminator**: 評估 motion 品質，生成 pseudo ground truth
3. **Confidence Selection**: 只選擇高信心的樣本進行訓練
4. **Data Augmentation**: 通過多次增強生成更穩定的 pseudo labels

## 架構組件

### 1. DomainAdaptiveGvhmrPL
- 繼承自原始的 `GvhmrPL`
- 添加了 domain adaptation 邏輯
- 支援 synthetic 和 real 數據的不同訓練策略

### 2. MotionDiscriminator
- LSTM-based 網路
- 輸入: SMPL pose parameters (69維)
- 輸出: confidence score [0,1]
- 支援 attention pooling

### 3. DomainAdaptiveDataModule
- 管理 synthetic 和 real datasets
- 提供交替採樣的 dataloader
- 支援多個 real datasets 的組合

## 使用方法

### 訓練命令

```bash
# 基本訓練
python tools/train.py exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw

# 調整參數
python tools/train.py exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw \
    domain_adaptation.confidence_threshold_start=0.7 \
    domain_adaptation.synthetic_weight=0.8 \
    domain_adaptation.real_weight=1.2

# 使用不同的 batch size
python tools/train.py exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw \
    data.loader_opts.train.batch_size=32 \
    pl_trainer.accumulate_grad_batches=4
```

### 測試

```bash
# 在 EMDB 和 3DPW 上測試
python tools/train.py global/task=gvhmr/test_3dpw_emdb \
    exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw \
    ckpt_path=path/to/your/checkpoint.ckpt
```

## 配置檔案結構

```
hmr4d/configs/
├── exp/gvhmr/domain_adaptive/
│   └── bedlam_emdb_3dpw.yaml          # 主要實驗配置
├── data/domain_adaptive/
│   └── bedlam_emdb_3dpw.yaml          # 數據配置
└── model/gvhmr/
    └── domain_adaptive_gvhmr_pl.yaml  # 模型配置
```

## 主要參數

### Domain Adaptation 參數

- `confidence_threshold_start/end`: 信心度閾值範圍
- `confidence_step`: 閾值更新步長
- `synthetic_weight/real_weight`: 合成/真實數據的 loss 權重
- `discriminator_weight`: discriminator loss 權重
- `num_augmentations`: 數據增強次數
- `alternating_ratio`: 交替訓練比例

### Training 參數

- `max_epochs: 300`: 更長的訓練時間
- `batch_size: 64`: 適中的 batch size
- `lr: 1e-4`: 較低的學習率
- `accumulate_grad_batches`: 梯度累積

## 監控與除錯

### TensorBoard 日誌

```bash
tensorboard --logdir outputs/domain_adaptive_v1/domain_adaptive_bedlam_emdb_3dpw/domain_adaptive_tb
```

### 關鍵指標

- `synthetic_loss`: 合成數據的 loss
- `real_loss`: 真實數據的 loss  
- `discriminator_*_loss`: discriminator 相關 losses
- `confidence_threshold`: 當前信心度閾值

## 故障排除

### 常見問題

1. **GPU 記憶體不足**
   - 降低 batch_size
   - 增加 accumulate_grad_batches
   - 使用 precision: 16-mixed

2. **訓練不穩定**
   - 調整 confidence_threshold_start
   - 降低 discriminator_weight
   - 增加 gradient_clip_val

3. **收斂慢**
   - 檢查數據平衡性
   - 調整 alternating_ratio
   - 確認 augmentation 策略

### Debug 選項

```bash
# 啟用詳細日誌
python tools/train.py exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw \
    pl_trainer.log_every_n_steps=10 \
    debug=True

# 限制數據量進行快速測試
python tools/train.py exp=gvhmr/domain_adaptive/bedlam_emdb_3dpw \
    pl_trainer.limit_train_batches=100 \
    pl_trainer.limit_val_batches=50
```

## 注意事項

1. **數據預處理**: 確保 BEDLAM、EMDB、3DPW 數據都正確預處理
2. **模型初始化**: 建議從預訓練的 GVHMR 模型開始
3. **硬體需求**: 需要足夠的 GPU 記憶體來同時載入兩種類型的數據
4. **訓練時間**: Domain adaptation 訓練通常需要更長時間

## 擴展

如需添加新的數據集或修改 discriminator 架構，請參考：

- `hmr4d/model/gvhmr/domain_adaptive_gvhmr_pl.py`
- `hmr4d/dataset/domain_adaptive/dual_datamodule.py`

並相應更新配置檔案。