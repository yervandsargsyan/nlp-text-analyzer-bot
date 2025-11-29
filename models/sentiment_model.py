# sentiment_async.py
import asyncio
import logging
from functools import lru_cache
from typing import List, Tuple, Optional

from langdetect import detect, DetectorFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------ Конфигурация ------------
DetectorFactory.seed = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Максимальное число конкурентных инференсов (регулируй по ресурсам)
# На CPU можно поставить 2-4, на GPU — 1-2 в зависимости от памяти.
MAX_CONCURRENT_INFERENCES = 2

# Семaфор для ограничения числа параллельных инференсов
_infer_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
# ---------------------------------------


def _get_device_index() -> int:
    """
    Возвращает device индекс для transformers.pipeline:
    -1 -> CPU
     0 -> первая CUDA-устройство
    """
    try:
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


# ------------- Помощники -------------
def detect_language(text: str) -> str:
    """
    Детектирование языка: 'en', 'ru' или 'other'
    (старательно сохраняем поведение старого кода).
    """
    try:
        lang = detect(text)
        if lang in ("en", "ru"):
            return lang
        return "other"
    except Exception:
        logger.exception("Language detection failed for text: %r", text)
        return "other"


def _label_to_numeric(label: str) -> float:
    """
    Универсальный мэппер меток в [0,1], аналогично ранее.
    """
    lab = (label or "").lower().strip()
    if not lab:
        return 0.5

    # stars: "5 stars", "4-star"
    if "star" in lab:
        for ch in lab:
            if ch.isdigit():
                stars = int(ch)
                return (stars - 1) / 4.0  # 1..5 -> 0..1

    if lab in ("negative", "neg", "negative)"):
        return 0.0
    if lab in ("neutral", "neu"):
        return 0.5
    if lab in ("positive", "pos"):
        return 1.0

    if lab.startswith("label_"):
        try:
            idx = int(lab.split("_", 1)[1])
            # LABEL_0 -> negative, LABEL_1 -> neutral, LABEL_2+ -> positive-ish
            if idx == 0:
                return 0.0
            if idx == 1:
                return 0.5
            return min(1.0, idx / 2.0)
        except Exception:
            pass

    if "pos" in lab:
        return 1.0
    if "neg" in lab:
        return 0.0

    return 0.5


# -------------------------------------


class ThreeModelsEnsemble:
    """
    Асинхронно-совместимый ансамбль на 3 моделях (или любое количество).
    Методы загрузки синхронны (pipeline) — вызываются в потоке.
    """

    def __init__(
        self,
        model_names: List[str],
        min_threshold: float = 0.34,
        max_threshold: float = 0.66,
        score_coeff: float = 1.0,
    ):
        self.model_names = model_names
        self.models: Optional[List] = None  # список pipeline объектов
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.score_coeff = score_coeff
        self.device = _get_device_index()

    def _load_models_sync(self):
        """
        Синхронная загрузка pipeline'ов (выполнится в отдельном треде).
        """
        if self.models is not None:
            return

        loaded = []
        for name in self.model_names:
            try:
                # Используем task "sentiment-analysis" — большинство моделей поддерживает
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
            # выполняем блокирующую загрузку в потоке
            await asyncio.to_thread(self._load_models_sync)

    async def classify_text_async(self, text: str) -> Tuple[str, float]:
        """
        Асинхронный вызов классификации: идёт через to_thread + семафор.
        Возвращает (sentiment_type, avg_score).
        Результаты вычисляются синхронно и кешируются (ниже).
        """
        await self._ensure_models_loaded()

        # Оборачиваем реальное sync-вычисление в общий кэшный синхронный слой.
        # lru_cache работает только для sync-функций, поэтому вызываем через to_thread.
        async with _infer_semaphore:
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
            # Если кто-то вызвал напрямую — попытка загрузить модели синхронно
            self._load_models_sync()

        weighted_scores = []
        for model in self.models:
            try:
                res = model(text)[0]  # pipeline возвращает list[dict]
                label = res.get("label", "")
                score = float(res.get("score", 1.0))
                numeric = _label_to_numeric(label)
                weighted_scores.append(numeric * (score ** self.score_coeff))
            except Exception:
                logger.exception("Model inference failed in ensemble for text: %r", text)
                # резервное значение — нейтральный вклад
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


# ------------- Singletons для трёх ансамблей -------------
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
            eng_model_names, min_threshold=0.15, max_threshold=0.73, score_coeff=1.0
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
            ru_model_names, min_threshold=0.29, max_threshold=0.66, score_coeff=0.5
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
            multi_model_names, min_threshold=0.12, max_threshold=0.53, score_coeff=1.0
        )
    return _multi_model


# ------------- Общая асинхронная точка входа -------------
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