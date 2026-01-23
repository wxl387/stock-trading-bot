"""
FinBERT-based financial sentiment analyzer.
Uses ProsusAI/finbert model locally on M2 MPS GPU.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid import overhead
_model = None
_tokenizer = None
_device = None


def _load_finbert():
    """Lazy-load FinBERT model and tokenizer."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Determine device
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        logger.info("FinBERT using MPS GPU acceleration")
    else:
        _device = torch.device("cpu")
        logger.info("FinBERT using CPU")

    logger.info("Loading FinBERT model (ProsusAI/finbert)...")
    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _model.to(_device)
    _model.eval()
    logger.info("FinBERT model loaded successfully")


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    Labels: positive, negative, neutral
    Score: -1 (bearish) to +1 (bullish)
    """

    def __init__(self):
        _load_finbert()

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Financial text to analyze (headline or summary)

        Returns:
            Dict with: sentiment (-1 to 1), label, confidence
        """
        import torch

        if not text or not text.strip():
            return {"sentiment": 0.0, "label": "neutral", "confidence": 0.0}

        # Truncate to max token length
        text = text[:512]

        inputs = _tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: positive, negative, neutral
        probs = probs.cpu().numpy()[0]
        positive, negative, neutral = probs[0], probs[1], probs[2]

        # Convert to single score: -1 to +1
        sentiment_score = float(positive - negative)

        # Determine label
        label_idx = np.argmax(probs)
        labels = ["positive", "negative", "neutral"]
        label = labels[label_idx]

        confidence = float(probs[label_idx])

        return {
            "sentiment": sentiment_score,
            "label": label,
            "confidence": confidence,
            "positive": float(positive),
            "negative": float(negative),
            "neutral": float(neutral),
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Batch sentiment analysis for efficiency.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch

        Returns:
            List of sentiment results
        """
        import torch

        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:512] if t else "" for t in batch]

            # Filter empty texts
            valid_indices = [j for j, t in enumerate(batch) if t.strip()]
            valid_texts = [batch[j] for j in valid_indices]

            if not valid_texts:
                results.extend([
                    {"sentiment": 0.0, "label": "neutral", "confidence": 0.0,
                     "positive": 0.0, "negative": 0.0, "neutral": 1.0}
                    for _ in batch
                ])
                continue

            inputs = _tokenizer(valid_texts, return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
            inputs = {k: v.to(_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = probs.cpu().numpy()

            # Map back to original batch positions
            batch_results = [
                {"sentiment": 0.0, "label": "neutral", "confidence": 0.0,
                 "positive": 0.0, "negative": 0.0, "neutral": 1.0}
                for _ in batch
            ]

            for idx, valid_idx in enumerate(valid_indices):
                p = probs[idx]
                positive, negative, neutral = float(p[0]), float(p[1]), float(p[2])
                label_idx = np.argmax(p)
                labels = ["positive", "negative", "neutral"]

                batch_results[valid_idx] = {
                    "sentiment": positive - negative,
                    "label": labels[label_idx],
                    "confidence": float(p[label_idx]),
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                }

            results.extend(batch_results)

        return results

    def analyze_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Analyze news articles and aggregate sentiment by date.

        Args:
            articles: List of article dicts with 'headline', 'summary', 'date'

        Returns:
            DataFrame with daily sentiment features
        """
        if not articles:
            return pd.DataFrame()

        # Combine headline + summary for analysis
        texts = []
        dates = []
        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline}. {summary}" if summary else headline
            texts.append(text)
            dates.append(article.get("date"))

        # Batch analyze
        sentiments = self.analyze_batch(texts)

        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "sentiment": [s["sentiment"] for s in sentiments],
            "confidence": [s["confidence"] for s in sentiments],
            "positive": [s["positive"] for s in sentiments],
            "negative": [s["negative"] for s in sentiments],
        })

        if df.empty:
            return pd.DataFrame()

        # Aggregate by date
        daily = df.groupby("date").agg(
            sentiment_score=("sentiment", "mean"),
            sentiment_confidence=("confidence", "mean"),
            article_count=("sentiment", "count"),
            sentiment_dispersion=("sentiment", "std"),
            positive_ratio=("positive", lambda x: (x > 0.5).mean()),
        ).reset_index()

        # Fill NaN dispersion (single article days) with 0
        daily["sentiment_dispersion"] = daily["sentiment_dispersion"].fillna(0)

        return daily
