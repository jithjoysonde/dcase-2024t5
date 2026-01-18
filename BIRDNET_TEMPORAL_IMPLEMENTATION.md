# BirdNet Temporal Embeddings Integration - Implementation Summary

## Overview
We've implemented **Option 1: Re-extract with Sliding Window** to convert BirdNet's single-vector embeddings into temporal sequences suitable for few-shot learning.

## Key Changes Made

### 1. **Feature Extraction (Sliding Windows)**
**File**: `birdnet/feature_extractor.py`
- **OLD**: Extract single 6522-dim embedding per audio file
- **NEW**: Extract temporal embeddings using 3-second sliding windows
  - Window size: 3.0 seconds (BirdNet input requirement)
  - Hop size: 3.0 seconds (non-overlapping for speed)
  - Output: `(num_windows, 6522)` array per audio file
  
**Example**: 1-hour audio file → ~1200 temporal windows → 1200×6522 embedding matrix

### 2. **Dataset Loader Updates**
**File**: `src/datamodules/components/birdnet_dataset.py`
- Updated `fps` (frames per second) from 200 to 0.333 (one 3s window per 3 seconds of audio)
- Fixed `select_segment()` to properly handle 2D embeddings `(time, 6522)`
- Added proper padding and tiling logic for temporal segments
- Removed assumptions about 1280-dim embeddings

### 3. **Encoder Model Updates**
**File**: `src/models/components/birdnet_encoder.py`
- Updated BirdNetEncoder to accept 6522-dim embeddings (was 1280)
- Temporal pooling applied to `(batch, time, 6522)` → `(batch, 6522)`
- Supports 'mean', 'max', or 'both' pooling strategies

### 4. **Configuration Updates**
**Files**: 
- `configs/train_birdnet.yaml`: `embedding_dim: 6522`
- `configs/model/birdnet_protonet.yaml`: `embedding_dim: 6522`

## Temporal Structure Explanation



## Performance Characteristics

| Metric | Value |
|--------|-------|
| Window Size | 3.0 seconds |
| Hop Size | 3.0 seconds (non-overlapping) |
| Embedding Dimension | 6522 |
| Temporal Resolution | ~0.33 Hz (one window per 3 seconds) |
| Model | BirdNet V2.4 TFLite |



## Extraction Script



```bash
cd /data/4joyson/bdnt
chmod +x extract_birdnet_sliding_window.sh
./extract_birdnet_sliding_window.sh
```

Or manually:

```bash
cd /data/4joyson/bdnt/birdnet
source /data/4joyson/dcase_t5/bin/activate

# Extract Training Set
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Training_Set \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place

# Extract Validation Set
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026 \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place
```

## Output Format

Each audio file gets a corresponding `_BDnet.npy` file:



## Training After Extraction

Once extraction completes:

```bash
cd /data/4joyson/bdnt
python train.py --config-name train_birdnet +exp_name="BirdNet"
```

**What happens during training**:
1. Dataloader loads `(num_windows, 6522)` embeddings
2. BirdNetEncoder applies temporal pooling: `(num_windows, 6522)` → `(1, 6522)`
3. Prototypical classifier processes 6522-dim vectors
4. PSDS evaluation unchanged (uses same metrics pipeline)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Files → BirdNet Extraction (3s windows)             │
│       ↓                                                      │
│  Temporal Embeddings (num_windows, 6522)                   │
│       ↓                                                      │
│  BirdNetDataset → Segment Selection → Random Crop          │
│       ↓                                                      │
│  Batch: (batch_size, seg_len, 6522)                        │
│       ↓                                                      │
│  BirdNetEncoder (Temporal Pooling) → (batch_size, 6522)    │
│       ↓                                                      │
│  Prototypical Classifier → 5-way Classification            │
│       ↓                                                      │
│  Contrastive Loss + PSDS Evaluation                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


---

**Status**: Ready for extraction and training. All components integrated and tested.
