"""
data/loader.py
==============
Loads and tokenizes agentic fine-tuning data.

Tool-call datasets:
    NousResearch/hermes-function-calling-v1
    glaiveai/glaive-function-calling-v2

Planning/reasoning datasets:
    openai/gsm8k
    TuringEnterprises/Turing-Open-Reasoning

Each sample is a full conversation tokenized with Qwen's chat template.
Labels = input_ids (next-token prediction). Padding positions = -100.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from utils.config import SystemConfig, QWEN_SPEC, LLAMA_SPEC

def load_tokenizer(model_family: str = "qwen"):
    spec = LLAMA_SPEC if model_family == "llama" else QWEN_SPEC
    tok = AutoTokenizer.from_pretrained(spec["hf_name"], trust_remote_code=True)
    return tok


class ConversationDataset(Dataset):
    """
    Each sample: tokenized conversation, truncated to max_seq_len.
    Labels = input_ids with padding positions masked to -100.
    """

    def __init__(self, texts, tok, max_seq_len):
        self.samples = []
        for text in texts:
            enc = tok(
                text,
                max_length    = max_seq_len,
                truncation    = True,
                padding       = "max_length",
                return_tensors= "pt",
            )
            ids     = enc["input_ids"].squeeze(0)
            mask    = enc["attention_mask"].squeeze(0)
            labels  = ids.clone()
            labels[mask == 0] = -100   # ignore padding in loss
            self.samples.append({"input_ids": ids, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def _safe_load(fn, label):
    try: return fn()
    except Exception as e: print(f"  [{label}] Skipped: {e}"); return None


def load_hermes(tok, max_seq_len, n=None):
    from datasets import load_dataset
    print("  [Hermes] Loading...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    if n: ds = ds.select(range(min(n, len(ds))))
    texts = []
    for row in ds:
        turns = [{"role": "user" if t.get("from")=="human" else "assistant",
                  "content": t.get("value","")}
                 for t in row.get("conversations", [])]
        if turns:
            texts.append(tok.apply_chat_template(turns, tokenize=False))
    print(f"  [Hermes] {len(texts):,} conversations")
    return ConversationDataset(texts, tok, max_seq_len)


def load_glaive(tok, max_seq_len, n=10000):
    from datasets import load_dataset
    print("  [Glaive] Loading...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    if n: ds = ds.select(range(min(n, len(ds))))
    texts = []
    for row in ds:
        chat = row.get("chat","").strip()
        if chat: texts.append(chat)
    print(f"  [Glaive] {len(texts):,} conversations")
    return ConversationDataset(texts, tok, max_seq_len)


def load_gsm8k(tok, max_seq_len, n=None):
    from datasets import load_dataset
    print("  [GSM8K] Loading...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n: ds = ds.select(range(min(n, len(ds))))
    texts = []
    for row in ds:
        turns = [{"role":"user","content": row["question"]},
                 {"role":"assistant","content": row["answer"]}]
        texts.append(tok.apply_chat_template(turns, tokenize=False))
    print(f"  [GSM8K] {len(texts):,} conversations")
    return ConversationDataset(texts, tok, max_seq_len)


def load_turing(tok, max_seq_len, n=None):
    from datasets import load_dataset
    print("  [Turing] Loading...")
    ds = load_dataset("TuringEnterprises/Turing-Open-Reasoning", split="train")
    if n: ds = ds.select(range(min(n, len(ds))))
    texts = []
    for row in ds:
        q = row.get("question",""); a = row.get("answer","")
        if q and a:
            turns = [{"role":"user","content":q},{"role":"assistant","content":a}]
            texts.append(tok.apply_chat_template(turns, tokenize=False))
    print(f"  [Turing] {len(texts):,} conversations")
    return ConversationDataset(texts, tok, max_seq_len)


def build_dataloaders(cfg: SystemConfig):
    tok        = load_tokenizer(cfg.model_family)
    sl         = cfg.training.max_seq_len
    bs         = cfg.training.batch_size
    n          = cfg.data.max_train_samples

    print("\n  Loading datasets...")
    all_ds = []
    for fn, label in [
        (lambda: load_hermes(tok, sl, n), "Hermes"),
        (lambda: load_glaive(tok, sl, min(n, 10000)), "Glaive"),
        (lambda: load_gsm8k(tok, sl, n), "GSM8K"),
        (lambda: load_turing(tok, sl, n), "Turing"),
    ]:
        ds = _safe_load(fn, label)
        if ds and len(ds) > 0:
            all_ds.append(ds)

    combined = ConcatDataset(all_ds) if len(all_ds) > 1 else all_ds[0]
    n_val    = max(1, int(len(combined) * cfg.data.val_split))
    n_train  = len(combined) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        combined, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels"   : torch.stack([b["labels"]    for b in batch]),
        }

    train_l = DataLoader(train_ds, batch_size=bs, shuffle=True,
                         drop_last=True, num_workers=2, collate_fn=collate)
    val_l   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                         drop_last=False, num_workers=2, collate_fn=collate)

    print(f"\n  Train: {len(train_ds):,} samples ({len(train_l)} batches)")
    print(f"  Val  : {len(val_ds):,} samples ({len(val_l)} batches)")
    return train_l, val_l
