import torch
import os
from torch.optim import AdamW  # Use PyTorch's AdamW
from tqdm import tqdm

def train_model(model, dataloader, epochs, device, save_path):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)  # No warning now
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for input_ids, attention_mask in loop:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
