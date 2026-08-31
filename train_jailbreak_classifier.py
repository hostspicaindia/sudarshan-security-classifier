"""
Sudarshan 1 - Jailbreak Prompt Classifier

Fine-tunes bert-base-multilingual-cased (110M params) to classify a prompt
as "jailbreak" (attempting to manipulate/bypass an LLM's instructions) or
"benign". Multilingual base model chosen so it has a real shot at Hindi/
Hinglish jailbreak attempts too, even though both datasets below are
English-only.

Two dataset options:
    jackhhao  - jackhhao/jailbreak-classification (v1). Small (1,306 rows),
                clean, pre-labeled "jailbreak"/"benign", fast to iterate on.
    necent    - Necent/llm-jailbreak-prompt-injection-dataset (v2). Much
                bigger (30+ public safety datasets aggregated, 1M-10M
                rows) and more diverse attack styles -- more robust, but
                needs real data-cleaning: it labels TWO separate things
                (prompt_harmful = dangerous topic, prompt_adversarial =
                manipulation/jailbreak technique) and many rows have
                these labels stubbed to null when not applicable. Only
                prompt_adversarial is used here -- that's the actual
                "is this trying to jailbreak the model" signal (a request
                can be harmful content without being an adversarial
                technique, or vice versa; those are different problems).
                Subsampled by default (--max-samples) since the full
                dataset would take unnecessarily long to fine-tune on for
                a v2 iteration -- 40K balanced rows is already ~30x more
                data and more source diversity than v1.

Usage:
    python train_jailbreak_classifier.py --dataset jackhhao
    python train_jailbreak_classifier.py --dataset necent
    python train_jailbreak_classifier.py --dataset necent --max-samples 100000
"""

import argparse

import numpy as np
from datasets import concatenate_datasets, load_dataset
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
SEED = 42

# v2 test showed the classifier missed every persona/roleplay-style jailbreak
# (DAN, "pretend you're my grandmother", fictional-framing) despite 96.6% eval
# F1 -- these sources are only ~7.5% of Necent's adversarial rows naturally,
# so a flat random subsample barely includes them. v3 oversamples them.
PRIORITY_SOURCES = {"WildJailbreak", "JailBreakV-28k"}


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


def load_jackhhao():
    """v1 dataset -- already has clean train/test splits and a single label."""
    ds = load_dataset("jackhhao/jailbreak-classification")

    def encode_labels(example):
        example["label"] = LABEL2ID[example["type"]]
        return example

    ds = ds.map(encode_labels)
    return ds.remove_columns(["type"])


def load_necent(max_samples: int, priority_fraction: float):
    """v3 dataset -- aggregated, needs filtering + stratified subsample + our
    own train/test split.

    prompt_adversarial is the label we want (1 = jailbreak/injection
    technique, 0 = not) -- matches LABEL2ID directly since it's already
    0/1. Rows where it's null (not applicable/not annotated) are dropped
    rather than guessed at.

    Adversarial rows are split into PRIORITY_SOURCES (persona/roleplay
    jailbreaks) and everything else, then sampled separately so priority
    sources hit priority_fraction of the adversarial half instead of their
    natural ~7.5% share.
    """
    print("loading dataset: Necent/llm-jailbreak-prompt-injection-dataset (this is large, may take a while)")
    raw = load_dataset("Necent/llm-jailbreak-prompt-injection-dataset", split="train")

    raw = raw.filter(lambda ex: ex["prompt_adversarial"] is not None and ex["prompt"])
    raw = raw.rename_column("prompt_adversarial", "label")
    raw = raw.select_columns(["prompt", "label", "source"])

    adv = raw.filter(lambda ex: ex["label"] == 1)
    benign = raw.filter(lambda ex: ex["label"] == 0)
    print(f"after filtering: {len(raw)} labeled rows ({len(adv)} jailbreak, {len(benign)} benign)")

    adv_target = max_samples // 2
    benign_target = max_samples - adv_target

    priority = adv.filter(lambda ex: ex["source"] in PRIORITY_SOURCES)
    other = adv.filter(lambda ex: ex["source"] not in PRIORITY_SOURCES)

    priority_target = min(int(adv_target * priority_fraction), len(priority))
    other_target = min(adv_target - priority_target, len(other))
    print(f"adversarial sample: {priority_target} priority (persona/roleplay) + {other_target} other")

    priority_sample = priority.shuffle(seed=SEED).select(range(priority_target))
    other_sample = other.shuffle(seed=SEED).select(range(other_target))
    benign_sample = benign.shuffle(seed=SEED).select(range(min(benign_target, len(benign))))

    combined = concatenate_datasets([priority_sample, other_sample, benign_sample])
    combined = combined.remove_columns(["source"]).shuffle(seed=SEED)

    split = combined.train_test_split(test_size=0.1, seed=SEED)
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["jackhhao", "necent"], default="necent")
    parser.add_argument("--max-samples", type=int, default=40000, help="cap on necent rows used (it's 1M-10M raw)")
    parser.add_argument(
        "--priority-fraction",
        type=float,
        default=0.3,
        help="v3: fraction of adversarial rows drawn from persona/roleplay sources (natural rate is ~7.5%%)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="checkpoints/jailbreak_classifier")
    args = parser.parse_args()

    if args.dataset == "jackhhao":
        ds = load_jackhhao()
    else:
        ds = load_necent(args.max_samples, args.priority_fraction)

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize(batch):
        return tokenizer(batch["prompt"], truncation=True, max_length=256, padding="max_length")

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["prompt"])
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
