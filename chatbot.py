from transformers import pipeline  # Import Hugging Face pipeline utility.


def main() -> None:  # Define the main function that runs the chatbot.
    model_name = "gpt2"  # Choose a small default text-generation model.
    print(f"Loading model: {model_name}")  # Tell the user model loading has started.

    generator = pipeline(  # Create a text-generation pipeline object.
        "text-generation",  # Select the text-generation task type.
        model=model_name,  # Tell pipeline which pretrained model to load.
    )  # Finish pipeline creation.

    print("Chatbot is ready. Type 'exit' to quit.")  # Show usage instructions.

    while True:  # Start an infinite loop for repeated chat turns.
        user_input = input("You: ").strip()  # Read and clean user input.
        if not user_input:  # Check for empty input.
            continue  # Skip empty messages and prompt again.
        if user_input.lower() == "exit":  # Check if user wants to stop.
            print("Goodbye!")  # Print closing message.
            break  # Exit the loop.

        prompt = f"User: {user_input}\nBot:"  # Build a simple chat-style prompt.
        output = generator(  # Generate text from the model.
            prompt,  # Provide the input prompt.
            max_new_tokens=60,  # Limit response length.
            do_sample=True,  # Enable sampling for varied replies.
            temperature=0.8,  # Control creativity (higher is more random).
            top_p=0.95,  # Use nucleus sampling.
            pad_token_id=generator.tokenizer.eos_token_id,  # Avoid padding warnings.
        )  # Finish generation call.

        generated_text = output[0]["generated_text"]  # Extract generated string.
        bot_reply = generated_text[len(prompt) :].strip()  # Keep only bot portion.
        print(f"Bot: {bot_reply}")  # Display bot response.


if __name__ == "__main__":  # Run main only when this file is executed directly.
    main()  # Start the chatbot program.
