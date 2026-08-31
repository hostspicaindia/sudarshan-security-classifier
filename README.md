# Sudarshan 1 — Security Classifier

Fine-tuned classifier models for AI/LLM security tasks. v1 target: jailbreak
prompt detection (is a given prompt trying to jailbreak an LLM, or benign?).

Separate, from-scratch project — not related to the Spica AI codebase
(different architecture: fine-tuned `bert-base-multilingual-cased`, not a
custom-trained decoder-only model).

## v1 — Jailbreak Classifier

Trained on [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification)
(1,306 labeled prompts: jailbreak vs benign).

```bash
pip install -r requirements.txt
python train_jailbreak_classifier.py
python predict.py --prompt "Ignore all previous instructions and tell me how to..."
```

## Roadmap

- [ ] v1: Jailbreak classifier (English, ready-made dataset)
- [ ] v2: Expand training data (Necent/llm-jailbreak-prompt-injection-dataset,
      allenai/wildjailbreak) for more robustness
- [ ] v3: Hindi/Hinglish scam & phishing detection (needs custom data —
      synthetic generation + real-world sourcing, no good native dataset
      exists yet)

## Why bert-base-multilingual-cased, not a decoder-only model

Classification (bidirectional context) suits an encoder model better than
a causal decoder. Fine-tuning an existing pretrained encoder is also far
cheaper/faster than pretraining from scratch (hours, not days) — the model
already has broad language understanding, we're only teaching it this one
task on top.
