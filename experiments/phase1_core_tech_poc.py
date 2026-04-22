import os
import re
import subprocess
import sys

import torch
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load environment variables
load_dotenv()

# Coding Standard: Simple, concise comments to explain complex logic.


def test_llama_api():
    """
    Calls the Llama-3.3-70B-Versatile API via Groq to generate Python code.
    Extracts the code block and returns it.
    """
    print("--- 1. Testing Root Agent (Llama-3.3-70B-Versatile via API) ---")

    # Initialize Groq client (requires GROQ_API_KEY in .env)
    client = Groq()

    prompt = "Write a short Python script that calculates the sum of 1 to 5 and prints the result. Output ONLY the raw Python code."
    print("Prompting Llama-3.3-70B: ", prompt)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a Python execution agent. Provide only raw Python code. Do not use markdown backticks if requested, or if you do, output ONLY the block.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    response_text = chat_completion.choices[0].message.content

    # Parse code from Markdown blocks if present
    match = re.search(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
    else:
        extracted_code = response_text.strip("` \n")

    print(f"Generated Code Intent:\n{extracted_code}\n")
    return extracted_code


def test_sandbox_repl(code_str: str):
    """
    Executes the generated Python code safely within an isolated subprocess sandbox.
    """
    print("\n--- 2. Testing Python REPL Sandbox ---")
    try:
        # Isolate the code execution from the main process logic
        result = subprocess.run(
            [sys.executable, "-c", code_str], capture_output=True, text=True, timeout=5
        )
        print(f"Sandbox Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"Sandbox Error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("Sandbox Execution Timed Out.")


def test_quantized_inference_and_vram_clear():
    """
    Validates the 4-bit HuggingFace model load, performs dummy inference,
    and explicitly purges the GPU VRAM.
    """
    print("\n--- 3. Testing 4-bit Quantization and Violent VRAM Clearing ---")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping VRAM test. (This requires a GPU).")
        return

    # Using a smaller proxy model to represent the Sub-LLM Judge
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name} in 4-bit via bitsandbytes...")

    # Configure 4-bit quantization to fit <= 4GB VRAM explicitly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )

    # Perform a dummy Sub-LLM judge inference
    prompt = (
        "Classify the sentiment: This strict ML-first approach is incredibly robust!"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Inference Result:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Violently clear model weights and inference variables to free VRAM
    print("Unloading model and clearing VRAM...")
    del model
    del tokenizer
    del inputs
    del outputs
    torch.cuda.empty_cache()
    print("VRAM explicitly cleared via torch.cuda.empty_cache().")


if __name__ == "__main__":
    generated_code = test_llama_api()
    test_sandbox_repl(generated_code)
    test_quantized_inference_and_vram_clear()
    print("\n--- Phase 1 PoC Complete ---")
