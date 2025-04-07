import os
import json
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP model")
    parser.add_argument("--config", type=str, default="configs/training_config.json", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    return parser.parse_args()

# Helper class for nested dictionaries
class AttrDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, AttrDict(value) if isinstance(value, dict) else value)
    
    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict(data)

# Configuration class
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

# Dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar = CIFAR10(root=root, train=train, download=True, transform=transform)
        
        # Load questions
        questions_path = os.path.join(root, 'cifar10_image_analysis.json')
        try:
            with open(questions_path, 'r') as f:
                self.questions_data = json.load(f)
                self.questions = self.questions_data['questions']
        except FileNotFoundError:
            # Default questions if file not found
            self.questions = [
                "What is the main object in this image?",
                "What is the dominant color in this image?",
                "What category does this image belong to?",
                "Describe this image in one sentence.",
                "What is the background of this image?"
            ]
            
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        
        # Get a random question
        question = random.choice(self.questions)
        
        # Create a simple answer based on CIFAR10 classes
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        answer = f"This is a {classes[label]}."
        
        return image, question, answer, label

# SigLIP Model
class SigLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load the full CLIP model
        self.model = CLIPModel.from_pretrained(config.model.vision_encoder)
        
        # Instead of using LoRA, we'll just make specific layers trainable
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Then unfreeze specific layers in the text model
        target_modules = config.lora.target_modules
        for name, param in self.model.text_model.named_parameters():
            if any(target in name for target in target_modules):
                param.requires_grad = True
                
        logging.info(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
            
    def forward(self, images, text_inputs):
        # Use the CLIP model's forward method without return_loss
        outputs = self.model(
            pixel_values=images,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        
        # Extract image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
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