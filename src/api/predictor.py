from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class PredictorConfig:
    model_path: str
    tokenizer_name: str
    threshold: float = 0.5
    max_length: int = 64


class SentimentPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path)
        self.model.eval()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    @staticmethod
    def _light_clean(text: str) -> str:
        return text.strip()

    def predict_one(self, text: str) -> Dict[str, Any]:
        cleaned = self._light_clean(text)

        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()
            probs = self._softmax(logits)[0]

        proba_neg = float(probs[0])
        proba_pos = float(probs[1])

        pred_label = int(np.argmax(probs))
        pred_text = "positif" if pred_label == 1 else "negatif"
        bad_buzz = bool(proba_neg >= self.cfg.threshold)

        return {
            "tweet": text,
            "tweet_clean": cleaned,
            "proba_pos": proba_pos,
            "proba_neg": proba_neg,
            "pred_label": pred_label,
            "pred_text": pred_text,
            "threshold": self.cfg.threshold,
            "bad_buzz": bad_buzz,
        }