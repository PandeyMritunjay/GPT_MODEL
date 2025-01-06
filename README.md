# GPT_MODEL

Link to the Google Colab - [GPT Model Google Colab](https://colab.research.google.com/drive/1Sem2Qwry8Bc4WgvW9Y9tsla_1NlI1NjX?usp=sharing)
###### Note- To run model better, change the hardware accelerator form CPU to T4 GPU.

## Key Steps:
- **Dataset Preparation**: Preprocess the dataset using a text file and load it with a custom `TextDataset` class. Already there, just used column of the data, as our custom dataset to train the Language model.
- **Model Training**: Fine-tune a pre-trained GPT-2 model using PyTorch, optimizing it with the AdamW optimizer.
- **Text Generation**: Generate text by feeding prompts into the trained GPT-2 model.
- **Model Saving**: Save the fine-tuned model’s weights to a specified path using PyTorch.
- **Error Handling and Debugging**: Address warnings and errors in generation configurations and Git issues.

## Tools Used:
- **Hugging Face Transformers**: For GPT-2 model implementation and tokenization.
- **PyTorch**: For model training, optimization, and saving the model weights.
- **TQDM**: For progress tracking during model training.
- **Python**: For dataset handling, training loops, and text generation.

## Challenges:
- **Memory Management**: Efficiently handling large datasets and training a transformer model.
- **Text Tokenization**: Handling text tokenization properly to prevent errors during training and generation.
- **Training Efficiency**: Optimizing model training and managing resources on available hardware.
- **Model Saving**: Saving large models to disk efficiently while ensuring compatibility.
- **GitHub File Size Limits**: Use Git LFS for large `.pth` files.
- **Error Debugging**: Resolve issues with model settings and configuration.

# Mritunjay Pandey
