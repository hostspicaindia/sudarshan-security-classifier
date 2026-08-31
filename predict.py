"""
Sudarshan 1 - Jailbreak Classifier Inference

Usage:
    python predict.py --prompt "Ignore all previous instructions and..."
    python predict.py --checkpoint checkpoints/jailbreak_classifier/final --prompt "What's the weather today?"
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/jailbreak_classifier/final")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    label_id = probs.argmax().item()
    label = model.config.id2label[label_id]
    confidence = probs[label_id].item()

    print(f"\nPrompt: {args.prompt}")
    print(f"Prediction: {label} ({confidence:.1%} confidence)")
    print(f"Full scores: benign={probs[0].item():.1%}, jailbreak={probs[1].item():.1%}")


if __name__ == "__main__":
    main()
