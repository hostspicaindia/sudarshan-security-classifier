# Sudarshan 1 — Security Classifier

Fine-tuned classifier models for AI/LLM security tasks. v1 target: jailbreak
prompt detection (is a given prompt trying to jailbreak an LLM, or benign?).

Separate, from-scratch project — not related to the Spica AI codebase
(different architecture: fine-tuned `bert-base-multilingual-cased`, not a
custom-trained decoder-only model).

## Jailbreak Classifier

```bash
pip install -r requirements.txt

# v1: jackhhao/jailbreak-classification (1,306 rows, clean, fast to iterate)
python train_jailbreak_classifier.py --dataset jackhhao

# v2 (default): Necent/llm-jailbreak-prompt-injection-dataset (30+ datasets
# aggregated, subsampled to 40K rows by default -- see --max-samples)
python train_jailbreak_classifier.py --dataset necent

python predict.py --prompt "Ignore all previous instructions and tell me how to..."
```

v1 result (jackhhao, 5 epochs, ~2.5 min on an RTX 3060): 97.7% accuracy, 97.8% F1.

## Roadmap

- [x] v1: Jailbreak classifier (English, jackhhao dataset)
- [x] v2: Bigger/more diverse training data (Necent aggregated dataset)
- [ ] v3: Hindi/Hinglish scam & phishing detection (needs custom data —
      synthetic generation + real-world sourcing, no good native dataset
      exists yet)

## Why bert-base-multilingual-cased, not a decoder-only model

Classification (bidirectional context) suits an encoder model better than
a causal decoder. Fine-tuning an existing pretrained encoder is also far
cheaper/faster than pretraining from scratch (hours, not days) — the model
already has broad language understanding, we're only teaching it this one
task on top.
