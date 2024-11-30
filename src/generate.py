import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(model_path, tokenizer_name, prompt, max_length=50, device="cpu"):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    model = GPT2LMHeadModel.from_pretrained(tokenizer_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
