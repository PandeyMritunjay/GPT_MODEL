from transformers import GPT2LMHeadModel

def get_model(model_name="gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model
