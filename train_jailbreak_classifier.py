"""
Sudarshan 1 - Jailbreak Prompt Classifier (v1)

Fine-tunes bert-base-multilingual-cased (110M params) on jackhhao/
jailbreak-classification to classify a prompt as "jailbreak" or "benign".
Multilingual base model chosen so it has a real shot at Hindi/Hinglish
jailbreak attempts too, even though this v1 dataset is English-only.

Usage:
    python train_jailbreak_classifier.py
    python train_jailbreak_classifier.py --epochs 8 --batch-size 8
"""

import argparse

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-multilingual-cased"
LABEL2ID = {"benign": 0, "jailbreak": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="checkpoints/jailbreak_classifier")
    args = parser.parse_args()

    print("loading dataset: jackhhao/jailbreak-classification")
    ds = load_dataset("jackhhao/jailbreak-classification")

    def encode_labels(example):
        example["label"] = LABEL2ID[example["type"]]
        return example

    ds = ds.map(encode_labels)

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize(batch):
        return tokenizer(batch["prompt"], truncation=True, max_length=256, padding="max_length")

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["prompt", "type"])
    ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nfinal evaluation on test set:")
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nmodel saved to {final_dir}")


if __name__ == "__main__":
    main()
