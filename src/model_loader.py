import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model path
MODEL_PATH = "model"  # Adjust if using Hugging Face Hub or cloud storage

def load_model():
    """Load the fine-tuned GPT model and tokenizer."""
    print("Loading model and tokenizer...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load model with proper device handling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)  # Move model to GPU if available

    print(f"Model loaded on {device}.")
    
    return model, tokenizer, device

# Test the loader
if __name__ == "__main__":
    model, tokenizer, device = load_model()
    print("Model and tokenizer successfully loaded.")
