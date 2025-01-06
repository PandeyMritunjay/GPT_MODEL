import os

import torch
from src.preprocess import get_dataloader
from src.model import get_model
from src.train import train_model
from src.generate import generate_text
from transformers import GPT2Tokenizer

def main():
    # Paths
    dataset_path = "./data/dataset.txt"
    model_save_path = './model/my_model.pth'
    tokenizer_name = "gpt2"
    
    # Prepare tokenizer and dataloader
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    dataloader = get_dataloader(dataset_path, tokenizer, batch_size=8, max_length=512)
    
    # Load and train model
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, dataloader, epochs=10, device=device, save_path=model_save_path)
    
    # Generate text
    prompt = "Once upon a time"
    generated_text = generate_text(model_save_path, tokenizer_name, prompt, device=device)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
