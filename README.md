# Vision-Language Model Training with SigLIP and Phi-3

This project implements a two-stage training process for creating a vision-language model:
1. Train a SigLIP model on CIFAR10 for vision-text alignment
2. Use the trained SigLIP model with Phi-3 to create a vision-language model using qLoRA

## Project Structure

```
era23_gpu_project/
├── configs/
│   └── training_config.json    # Configuration for both training stages
├── scripts/
│   └── runpod_train.sh        # Main training script
├── src/
│   ├── train_siglip.py        # SigLIP training implementation
│   └── train_phi3_vlm.py      # Phi-3 VLM training implementation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Requirements

- NVIDIA GPU with CUDA support (tested on A40)
- Python 3.8+
- PyTorch 2.1.0
- Transformers 4.35.2
- Other dependencies listed in requirements.txt

## Setup and Training

1. Upload the project to RunPod:
   ```bash
   # On your local machine
   zip -r era23_gpu_project.zip era23_gpu_project/
   # Upload to RunPod
   ```

2. On RunPod, extract and prepare:
   ```bash
   unzip era23_gpu_project.zip
   cd era23_gpu_project
   chmod +x scripts/runpod_train.sh
   ```

3. Start training:
   ```bash
   ./scripts/runpod_train.sh
   ```

The script will:
1. Create necessary directories in /workspace
2. Verify CUDA setup
3. Train SigLIP model first
4. Use the best SigLIP checkpoint to train Phi-3 VLM

## Training Configuration

The training parameters can be modified in `configs/training_config.json`:

```json
{
    "model": {
        "vision_encoder": "openai/clip-vit-base-patch32",
        "text_encoder": "openai/clip-vit-base-patch32",
        "phi3_model": "microsoft/phi-3-mini-4k-instruct"
    },
    "data": {
        "train_batch_size": 32,
        "eval_batch_size": 16,
        "num_workers": 4,
        "image_size": 224
    },
    "training": {
        "siglip_epochs": 5,
        "vlm_epochs": 3,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 4,
        "fp16": true
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
}
```

## Output Structure

The training outputs will be saved in the following locations:

- `/workspace/checkpoints/siglip/`: SigLIP model checkpoints
  - `checkpoint-{epoch}.pt`: Regular checkpoints
  - `best_checkpoint.pt`: Best performing model

- `/workspace/results/`: Phi-3 VLM checkpoints
  - `checkpoint-{epoch}/`: Regular checkpoints with LoRA weights
  - `best_checkpoint/`: Best performing model with LoRA weights

## Monitoring

Both training stages output detailed logs:
- `siglip_training.log`: SigLIP training progress
- `phi3_vlm_training.log`: Phi-3 VLM training progress

The logs include:
- Loss values
- Training progress
- GPU utilization
- Error messages (if any)

## Notes

- The training uses mixed precision (FP16) by default for better memory efficiency
- SigLIP uses LoRA for the text encoder only
- Phi-3 uses qLoRA (4-bit quantization with LoRA) for efficient fine-tuning
- The image encoder from SigLIP is frozen during VLM training

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

# ERA23 GPU Project: SigLIP Training

This repository contains code for training a SigLIP (CLIP-based) vision-language model using PyTorch. The training was performed on a NVIDIA A40 GPU.

## Training Summary

- **Model**: CLIP (vision-language model)
- **Dataset**: CIFAR10
- **Image Size**: 224x224
- **Batch Size**: 32
- **Epochs**: 5
- **Training Time**: ~12 minutes
- **Final Loss**: 3.4653

## Training Log

```
[2025-04-06 13:38:18] Creating project structure...
[2025-04-06 13:38:18] Creating questions file...
[2025-04-06 13:38:18] Verifying GPU setup...
PyTorch version: 2.1.0+cu118
CUDA available: True
GPU device name: NVIDIA A40
GPU memory: 44.45 GB
[2025-04-06 13:38:20] Starting fixed SigLIP training...
2025-04-06 13:38:23,385 - INFO - Using data.image_size (224) for model.image_size
2025-04-06 13:38:23,385 - INFO - Using device: cuda
Files already downloaded and verified
2025-04-06 13:38:25,783 - INFO - Number of trainable parameters: 9455616
2025-04-06 13:38:26,119 - INFO - Starting epoch 1/5
Epoch 1:   0%|                                                                                                            | 0/1563 [00:00<?, ?it/s]2025-04-06 13:38:26,358 - INFO - Image shape: torch.Size([32, 3, 224, 224])
2025-04-06 13:38:26,358 - INFO - Input IDs shape: torch.Size([32, 77])
2025-04-06 13:38:26,358 - INFO - Attention mask shape: torch.Size([32, 77])
Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [02:21<00:00, 11.04it/s, loss=2.77]
2025-04-06 13:40:47,657 - INFO - Epoch 1 - Training loss: 3.4654
2025-04-06 13:40:48,887 - INFO - New best model saved with loss: 3.4654
2025-04-06 13:40:48,887 - INFO - Starting epoch 2/5
Epoch 2:   0%|                                                                                                            | 0/1563 [00:00<?, ?it/s]2025-04-06 13:40:49,178 - INFO - Image shape: torch.Size([32, 3, 224, 224])
2025-04-06 13:40:49,178 - INFO - Input IDs shape: torch.Size([32, 77])
2025-04-06 13:40:49,178 - INFO - Attention mask shape: torch.Size([32, 77])
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [02:21<00:00, 11.04it/s, loss=2.77]
2025-04-06 13:43:10,470 - INFO - Epoch 2 - Training loss: 3.4653
2025-04-06 13:43:11,683 - INFO - New best model saved with loss: 3.4653
2025-04-06 13:43:11,684 - INFO - Starting epoch 3/5
Epoch 3:   0%|                                                                                                            | 0/1563 [00:00<?, ?it/s]2025-04-06 13:43:12,177 - INFO - Image shape: torch.Size([32, 3, 224, 224])
2025-04-06 13:43:12,177 - INFO - Input IDs shape: torch.Size([32, 77])
2025-04-06 13:43:12,177 - INFO - Attention mask shape: torch.Size([32, 77])
Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [02:22<00:00, 10.99it/s, loss=2.77]
2025-04-06 13:45:33,934 - INFO - Epoch 3 - Training loss: 3.4653
2025-04-06 13:45:35,028 - INFO - New best model saved with loss: 3.4653
2025-04-06 13:45:35,029 - INFO - Starting epoch 4/5
Epoch 4:   0%|                                                                                                            | 0/1563 [00:00<?, ?it/s]2025-04-06 13:45:35,470 - INFO - Image shape: torch.Size([32, 3, 224, 224])
2025-04-06 13:45:35,471 - INFO - Input IDs shape: torch.Size([32, 77])
2025-04-06 13:45:35,471 - INFO - Attention mask shape: torch.Size([32, 77])
Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [02:22<00:00, 11.00it/s, loss=2.77]
2025-04-06 13:47:57,139 - INFO - Epoch 4 - Training loss: 3.4653
2025-04-06 13:47:57,682 - INFO - Starting epoch 5/5
Epoch 5:   0%|                                                                                                            | 0/1563 [00:00<?, ?it/s]2025-04-06 13:47:58,065 - INFO - Image shape: torch.Size([32, 3, 224, 224])
2025-04-06 13:47:58,066 - INFO - Input IDs shape: torch.Size([32, 77])
2025-04-06 13:47:58,066 - INFO - Attention mask shape: torch.Size([32, 77])
Epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [02:20<00:00, 11.09it/s, loss=2.77]
2025-04-06 13:50:18,617 - INFO - Epoch 5 - Training loss: 3.4653
2025-04-06 13:50:19,693 - INFO - New best model saved with loss: 3.4653
2025-04-06 13:50:19,693 - INFO - SigLIP training completed!
[2025-04-06 13:50:20] Fixed training completed!
```

## Model Information

- **Base Model**: `openai/clip-vit-base-patch32`
- **Fine-tuning Method**: Selective layer unfreezing (targeting key layers in the text encoder)
- **Trainable Parameters**: 9,455,616
- **Training Strategy**: Contrastive Learning with questions about images
- **Loss Function**: Cross-entropy on image-text similarity matrix

## Files and Directory Structure

```
era23_gpu_project/
├── configs/
│   └── training_config.json     # Training configuration
├── scripts/
│   ├── run_fixed_training.sh    # Training script
│   ├── test_model.sh            # Script to test the model (with visualization)
│   ├── test_model_runpod.sh     # Script to test on RunPod (no visualization)
│   └── download_test_images.sh  # Script to download test images
├── src/
│   ├── fixed_train_siglip.py    # Main training code
│   ├── inference.py             # Model inference code (with visualization)
│   └── inference_no_display.py  # Model inference code (no visualization)
├── data/
│   ├── cifar10/                 # Dataset and questions
│   └── test_images/             # Test images for inference
├── checkpoints/                 # Saved model checkpoints
└── README.md                    # This file
```

## Saved Checkpoints

The trained model checkpoints are saved in the `/data/checkpoints` directory:
- `/data/checkpoints/checkpoint-1.pt` through `/data/checkpoints/checkpoint-5.pt` (one for each epoch)
- `/data/checkpoints/best_checkpoint.pt` (best model based on training loss)

## Testing the Model

### On RunPod (no display)

1. Download test images:
   ```bash
   cd /era23_gpu_project
   chmod +x scripts/download_test_images.sh
   ./scripts/download_test_images.sh
   ```

2. Test the model on an image:
   ```bash
   chmod +x scripts/test_model_runpod.sh
   ./scripts/test_model_runpod.sh /data/test_images/car.jpg
   ```

3. View the results:
   ```bash
   cat /data/test_results_car.txt
   ```

### On Local Machine (with display)

1. Download the checkpoint from RunPod to your local machine.

2. Install the required dependencies:
   ```bash
   pip install torch torchvision transformers PIL matplotlib
   ```

3. Run the inference script:
   ```bash
   python src/inference.py --checkpoint /path/to/checkpoint.pt --image /path/to/image.jpg
   ```

## Usage in Python Code

To use the trained model for inference:

```python
import torch
from transformers import CLIPModel, CLIPTokenizer

# Load the model
model_path = "/data/checkpoints/best_checkpoint.pt"
checkpoint = torch.load(model_path)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Prepare inputs
image = ...  # Load and preprocess your image
text = "What is in this image?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Run inference
with torch.no_grad():
    outputs = model(pixel_values=image.unsqueeze(0), **inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    
    # Calculate similarity
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    print(f"Similarity score: {similarity.item()}")
``` 