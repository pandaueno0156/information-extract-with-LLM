import argparse
import time
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def batch_generate(tokenizer, model, prompts, max_new_tokens=20):
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2
        )
    elapsed = time.time() - start
    decoded_outputs = tokenizer.batch_decode(outputs[:, tokenized["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded_outputs, elapsed


def main(args):
    # Load model
    tokenizer, model = load_model(args.model_path)

    # Load test data
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Inference
    batched_preds = []
    inference_times = []
    for i in range(0, len(df), args.batch_size):
        batch = df[args.prompt_column].iloc[i:i+args.batch_size].tolist()
        outputs, t = batch_generate(tokenizer, model, batch, max_new_tokens=args.max_new_tokens)
        batched_preds.extend(outputs)
        inference_times.extend([t / len(batch)] * len(batch))

    df['y_pred'] = batched_preds
    df['inference_time'] = inference_times

    # Save result
    df.to_json(args.output_json, orient="records", force_ascii=False, indent=2)
    print(f"Inference completed. Results saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference with Qwen Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the Hugging Face model')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save output JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max_new_tokens', type=int, default=20, help='Maximum new tokens to generate')
    parser.add_argument('--prompt_column', type=str, default='prompt', help='Name of the prompt column in the input JSON')

    args = parser.parse_args()
    main(args)
