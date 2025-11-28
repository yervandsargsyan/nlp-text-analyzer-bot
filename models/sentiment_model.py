from langdetect import detect, DetectorFactory
from transformers import pipeline

DetectorFactory.seed = 42

def detect_language(text):
    try:
        lang = detect(text)
        if lang in ("en", "ru"):
            return lang
        else:
            return "other"
    except:
        return "other"


class ThreeModelsEnsemble:
    def __init__(self, model_names, min_threshold=0.34, max_threshold=0.66, score_coeff=1.0):
        self.model_names = model_names
        self.models = None
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.score_coeff = score_coeff

    def _load_models(self):
        if self.models is None:
            self.models = [pipeline("sentiment-analysis", model=name) for name in self.model_names]

    def classify_text(self, text):
        self._load_models()
        weighted_scores = []

        for model in self.models:
            result = model(text)[0]
            label = result['label'].lower()
            score = result['score']

            if 'star' in label:
                stars = int(label[0])
                numeric = (stars - 1) / 4
            else:
                if label in ('negative', 'neg', 'LABEL_0'):
                    numeric = 0.0
                elif label in ('neutral', 'neu', 'LABEL_1'):
                    numeric = 0.5
                else:
                    numeric = 1.0
            weighted_scores.append(numeric * score ** self.score_coeff)

        avg_score = sum(weighted_scores) / len(weighted_scores)
        if avg_score <= self.min_threshold:
            sentiment_type = "negative"
        elif avg_score >= self.max_threshold:
            sentiment_type = "positive"
        else:
            sentiment_type = "neutral"

        return sentiment_type, avg_score

_eng_model = None
_ru_model = None
_multi_model = None

def get_eng_model():
    global _eng_model
    if _eng_model is None:
        eng_model_names = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "finiteautomata/bertweet-base-sentiment-analysis"
        ]
        _eng_model = ThreeModelsEnsemble(
            eng_model_names,
            min_threshold=0.15,
            max_threshold=0.73
        )
    return _eng_model

def get_ru_model():
    global _ru_model
    if _ru_model is None:
        ru_model_names = [
            "blanchefort/rubert-base-cased-sentiment",
            "r1char9/rubert-base-cased-russian-sentiment",
            "seara/rubert-tiny2-russian-sentiment"
        ]
        _ru_model = ThreeModelsEnsemble(
            ru_model_names,
            min_threshold=0.29,
            max_threshold=0.66,
            score_coeff=0.5
        )
    return _ru_model

def get_multi_model():
    global _multi_model
    if _multi_model is None:
        multi_model_names = [
            "clapAI/roberta-base-multilingual-sentiment",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        ]
        _multi_model = ThreeModelsEnsemble(
            multi_model_names,
            min_threshold=0.12,
            max_threshold=0.53
        )
    return _multi_model

def get_sentiment(text):
    lang = detect_language(text)
    if lang == "en":
        return get_eng_model().classify_text(text)
    elif lang == "ru":
        return get_ru_model().classify_text(text)
    else:
        return get_multi_model().classify_text(text)
