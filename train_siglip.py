import os
import sys
import logging
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup, CLIPModel, CLIPProcessor, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('siglip_training.log')
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SigLIP model on CIFAR10')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    return parser.parse_args()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
            return data
        result = AttrDict()
        for key, value in data.items():
            result[key] = AttrDict.from_nested_dict(value)
        return result

class SigLIPConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        nested_config = AttrDict.from_nested_dict(config)
        self.training = nested_config.training
        self.model = nested_config.model
        self.lora = nested_config.lora
        self.data = nested_config.data
        
        # Ensure image_size exists in model config
        if not hasattr(self.model, 'image_size') and hasattr(self.data, 'image_size'):
            self.model.image_size = self.data.image_size
            logging.info(f"Using data.image_size ({self.data.image_size}) for model.image_size")

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar = CIFAR10(root=root, train=train, download=True, transform=transform)
        self.transform = transform
        
        # Load questions
        with open(os.path.join(root, 'cifar10_image_analysis.json'), 'r') as f:
            self.questions = json.load(f)['questions']
            
        # CIFAR10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create templates for answers
        self.templates = [
            "The main object in this image is a {}.",
            "The main object is {}.",
            "This image was taken {}.",
            "The background of this image is {}.",
            "There are {} people in this image.",
            "The lighting condition is {}.",
            "The image is {}.",
            "The composition of this image is {}.",
            "There are {} text or numbers visible.",
            "The overall mood of this image is {}."
        ]
        
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        class_name = self.class_names[label]
        
        # Select random question
        q_idx = np.random.randint(0, len(self.questions))
        question = self.questions[q_idx]
        
        # Generate answer based on the question
        if q_idx == 0:  # What is the main object
            answer = self.templates[q_idx].format(class_name)
        elif q_idx == 1:  # Color
            colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray']
            answer = self.templates[q_idx].format(np.random.choice(colors))
        elif q_idx == 2:  # Indoor/outdoor
            locations = ['indoors', 'outdoors']
            answer = self.templates[q_idx].format(np.random.choice(locations))
        elif q_idx == 3:  # Background
            backgrounds = ['blurry', 'clear', 'natural', 'plain', 'textured']
            answer = self.templates[q_idx].format(np.random.choice(backgrounds))
        elif q_idx == 4:  # People
            num_people = ['no', 'some', 'many']
            answer = self.templates[q_idx].format(np.random.choice(num_people))
        elif q_idx == 5:  # Lighting
            lighting = ['bright', 'dim', 'natural', 'artificial']
            answer = self.templates[q_idx].format(np.random.choice(lighting))
        elif q_idx == 6:  # Clear/blurry
            clarity = ['clear', 'slightly blurry', 'very clear']
            answer = self.templates[q_idx].format(np.random.choice(clarity))
        elif q_idx == 7:  # Composition
            compositions = ['centered', 'balanced', 'asymmetrical']
            answer = self.templates[q_idx].format(np.random.choice(compositions))
        elif q_idx == 8:  # Text/numbers
            text = ['no', 'some', 'no visible']
            answer = self.templates[q_idx].format(np.random.choice(text))
        elif q_idx == 9:  # Mood
            moods = ['neutral', 'calm', 'energetic', 'playful']
            answer = self.templates[q_idx].format(np.random.choice(moods))
        
        return image, question, answer, label

class SigLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Import the CLIPModel class directly
        from transformers import CLIPModel, CLIPProcessor
        
        # Load the full CLIP model (includes both vision and text encoders)
        self.model = CLIPModel.from_pretrained(config.model.vision_encoder)
        self.processor = CLIPProcessor.from_pretrained(config.model.vision_encoder)
        
        # Apply LoRA to the text encoder
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type="FEATURE_EXTRACTION"
        )
        self.model.text_model = get_peft_model(self.model.text_model, lora_config)
        
        # Freeze the image encoder (we'll only train the text encoder with LoRA)
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
            
        logging.info(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
            
    def forward(self, images, text_inputs):
        # Use the CLIP model's forward method directly
        outputs = self.model(
            pixel_values=images,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            return_loss=True
        )
        
        # Extract image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # The model already normalizes the features, but let's ensure they're normalized
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        logits = torch.matmul(image_features, text_features.transpose(0, 1))
        
        return logits, image_features, text_features

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, epoch, config):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        try:
            images, questions, answers, labels = batch
            
            # Move images to device
            images = images.to(device)
            
            # Tokenize questions with proper padding and truncation
            question_inputs = tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=77,  # CLIP's default max length
                return_tensors="pt"
            ).to(device)
            
            # Debug info
            if step == 0:
                logging.info(f"Image shape: {images.shape}")
                logging.info(f"Input IDs shape: {question_inputs.input_ids.shape}")
                logging.info(f"Attention mask shape: {question_inputs.attention_mask.shape}")
            
            # Forward pass with question inputs
            logits, _, _ = model(images, question_inputs)
            
            # Compute loss (contrastive)
            targets = torch.arange(len(logits)).to(device)
            loss = F.cross_entropy(logits, targets)
            
            # Gradient accumulation
            loss = loss / config.training.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.training.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * config.training.gradient_accumulation_steps})
        
        except Exception as e:
            logging.error(f"Error at step {step}: {e}")
            logging.error(f"Batch info: images={images.shape if images is not None else None}, "
                         f"questions={len(questions) if questions is not None else None}")
            # Continue with next batch
            continue
    
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = SigLIPConfig(args.config)
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.model.image_size, config.model.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    train_dataset = CIFAR10Dataset(root=args.data_dir, train=True, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.data.train_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    # Initialize model
    model = SigLIPModel(config).to(device)
    
    # Initialize tokenizer from the same CLIP model
    tokenizer = CLIPTokenizer.from_pretrained(config.model.vision_encoder)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * config.training.siglip_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Enable mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.training.siglip_epochs):
        logging.info(f"Starting epoch {epoch+1}/{config.training.siglip_epochs}")
        
        train_loss = train_one_epoch(model, tokenizer, train_dataloader, optimizer, scheduler, device, epoch, config)
        logging.info(f"Epoch {epoch+1} - Training loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, best_checkpoint_path)
            logging.info(f"New best model saved with loss: {best_loss:.4f}")
    
    logging.info("SigLIP training completed!")

if __name__ == "__main__":
    main() 