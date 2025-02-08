import torch
from model_loader import load_model

# Load model and tokenizer
model, tokenizer, device = load_model()

def generate_response(prompt):
    """Generates a response from the fine-tuned GPT chatbot."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the chatbot function
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        bot_response = generate_response(user_input)
        print(f"Bot: {bot_response}")
