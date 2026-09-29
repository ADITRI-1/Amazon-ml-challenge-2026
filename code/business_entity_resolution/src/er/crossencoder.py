"""Cross-encoder over (S1 record, candidate record) pairs: encoder + mean pooling + linear head -> logit."""
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CE_MAX_LEN = 128


def pair_text(name: str, addr: str) -> str:
    return f"{name} | {addr}"


_LIG = {"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss", "º": "o", "°": "o"}


def fold_latin(s: str) -> str:
    """Strip diacritics from LATIN letters only (é→e, ç→c, à→a); leave Devanagari/Tamil/... untouched."""
    out = []
    for ch in s:
        if ch in _LIG:
            out.append(_LIG[ch])
        elif ord(ch) > 127 and "LATIN" in unicodedata.name(ch, ""):
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


class CrossEncoder(torch.nn.Module):
    def __init__(self, path, train=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        head = Path(path) / "ce_head.pt"
        if head.exists():
            self.head.load_state_dict(torch.load(head, map_location="cpu"))
        cfg = Path(path) / "text_cfg.json"
        self.fold = bool(json.load(open(cfg)).get("fold_latin")) if cfg.exists() else False  # stored with the model
        self.cuda().train(train)

    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.enc.save_pretrained(out); self.tok.save_pretrained(out)
        torch.save(self.head.state_dict(), out / "ce_head.pt")
        json.dump({"fold_latin": self.fold}, open(out / "text_cfg.json", "w"))


class PairDS(Dataset):
    def __init__(self, ta, tb, y, tok, fold=False):
        self.ta, self.tb, self.y, self.tok, self.fold = ta, tb, y, tok, fold

    def __len__(self):
        return len(self.ta)

    def __getitem__(self, i):
        return i

    def collate(self, idx):
        f = fold_latin if self.fold else (lambda x: x)
        enc = self.tok([f(self.ta[i]) for i in idx], [f(self.tb[i]) for i in idx], truncation=True,
                       max_length=CE_MAX_LEN, padding=True, return_tensors="pt")
        y = torch.tensor(self.y[idx] if self.y is not None else np.zeros(len(idx)), dtype=torch.float32)
        return enc["input_ids"], enc["attention_mask"], y


@torch.no_grad()
def ce_score(model, ta, tb, bs=1024, workers=16) -> np.ndarray:
    """logits for aligned text lists; batches are length-sorted for speed."""
    model.eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)], kind="stable")
    ta_s, tb_s = [ta[i] for i in order], [tb[i] for i in order]
    ds = PairDS(ta_s, tb_s, None, model.tok, fold=model.fold)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=ds.collate, prefetch_factor=4)
    out = []
    for ids, am, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids.cuda(non_blocking=True), am.cuda(non_blocking=True)).float().cpu().numpy())
    s = np.concatenate(out)
    res = np.empty_like(s)
    res[order] = s
    return res


class HFCrossEncoder(torch.nn.Module):
    """Native 1-logit sequence-classification reranker (e.g. fine-tuned bge-reranker-v2-m3 from blackwell_kit), fp16."""

    def __init__(self, path):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
        self.fold = False

    def forward(self, ids, am):
        return self.m(input_ids=ids, attention_mask=am).logits.squeeze(-1)
