# sentiment_async.py

import asyncio
from functools import lru_cache
from typing import List, Tuple, Optional
from utils.tools import detect_language, label_to_numeric
from utils.config import logger_config, get_device_index
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


logger, infer_semaphore = logger_config()

class ThreeModelsEnsemble:

    def __init__(
        self,
        model_names: List[str],
        min_threshold: float = 0.34,
        max_threshold: float = 0.66,
        score_coef: float = 1.0,
    ):
        self.model_names = model_names
        self.models: Optional[List] = None  # список pipeline объектов
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.score_coef = score_coef
        self.device = get_device_index()

    def _load_models_sync(self):
        """
        Синхронная загрузка pipeline'ов (выполнится в отдельном треде).
        """
        if self.models is not None:
            return

        loaded = []
        for name in self.model_names:
            try:
                p = pipeline("sentiment-analysis", model=name, device=self.device)
                loaded.append(p)
                logger.info("Loaded model %s (device=%s)", name, self.device)
            except Exception:
                logger.exception("Failed to load model %s", name)
        if not loaded:
            raise RuntimeError(f"No models could be loaded for {self.model_names}")
        self.models = loaded

    async def _ensure_models_loaded(self):
        """
        Асинхронный wrapper: загрузка моделей в фоне, если ещё не загружены.
        """
        if self.models is None:
            await asyncio.to_thread(self._load_models_sync)

    async def classify_text_async(self, text: str) -> Tuple[str, float]:
        """
        Асинхронный вызов классификации: идёт через to_thread + семафор.
        Возвращает (sentiment_type, avg_score).
        Результаты вычисляются синхронно и кешируются (ниже).
        """
        await self._ensure_models_loaded()

        async with infer_semaphore:
            return await asyncio.to_thread(self._classify_text_sync, text)

    @lru_cache(maxsize=4096)
    def _classify_text_sync(self, text: str) -> Tuple[str, float]:
        """
        Реальная синхронная логика: вызывает все модели по очереди,
        агрегирует результаты и возвращает итог.
        Помечена lru_cache для кэширования по тексту.
        ВНИМАНИЕ: _classify_text_sync выполняется внутри потока (to_thread).
        """
        if self.models is None:
            self._load_models_sync()

        weighted_scores = []
        for model in self.models:
            try:
                res = model(text)[0]  
                label = res.get("label", "")
                score = float(res.get("score", 1.0))
                numeric = label_to_numeric(label)
                weighted_scores.append(numeric * (score ** self.score_coef))
            except Exception:
                logger.exception("Model inference failed in ensemble for text: %r", text)
                weighted_scores.append(0.5)

        if not weighted_scores:
            avg_score = 0.5
        else:
            avg_score = sum(weighted_scores) / len(weighted_scores)

        if avg_score <= self.min_threshold:
            sentiment_type = "negative"
        elif avg_score >= self.max_threshold:
            sentiment_type = "positive"
        else:
            sentiment_type = "neutral"

        return sentiment_type, float(avg_score)


_eng_model: Optional[ThreeModelsEnsemble] = None
_ru_model: Optional[ThreeModelsEnsemble] = None
_multi_model: Optional[ThreeModelsEnsemble] = None


def get_eng_model() -> ThreeModelsEnsemble:
    global _eng_model
    if _eng_model is None:
        eng_model_names = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "finiteautomata/bertweet-base-sentiment-analysis",
        ]
        _eng_model = ThreeModelsEnsemble(
            eng_model_names, min_threshold=0.15, max_threshold=0.73, score_coef=1.0
        )
    return _eng_model


def get_ru_model() -> ThreeModelsEnsemble:
    global _ru_model
    if _ru_model is None:
        ru_model_names = [
            "blanchefort/rubert-base-cased-sentiment",
            "r1char9/rubert-base-cased-russian-sentiment",
            "seara/rubert-tiny2-russian-sentiment",
        ]
        _ru_model = ThreeModelsEnsemble(
            ru_model_names, min_threshold=0.29, max_threshold=0.66, score_coef=0.5
        )
    return _ru_model


def get_multi_model() -> ThreeModelsEnsemble:
    global _multi_model
    if _multi_model is None:
        multi_model_names = [
            "clapAI/roberta-base-multilingual-sentiment",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        ]
        _multi_model = ThreeModelsEnsemble(
            multi_model_names, min_threshold=0.12, max_threshold=0.53, score_coef=1.0
        )
    return _multi_model


async def get_sentiment(text: str) -> Tuple[str, float]:
    """
    Асинхронная функция для использования в боте.
    Выбирает ансамбль по языку и вызывает его.
    """
    lang = detect_language(text)
    if lang == "en":
        model = get_eng_model()
    elif lang == "ru":
        model = get_ru_model()
    else:
        model = get_multi_model()

    return await model.classify_text_async(text)
