import os
import torch
from flask import Flask, request, jsonify, render_template
from src.preprocess import get_dataloader
from src.model import get_model
from src.train import train_model
from src.generate import generate_text
from transformers import GPT2Tokenizer

app = Flask(__name__, template_folder="templates")

# Load model and tokenizer
model_save_path = './model/my_model.pth'
tokenizer_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
model = get_model()
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    generated_text = generate_text(model_save_path, tokenizer_name, prompt, device=device)
    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
