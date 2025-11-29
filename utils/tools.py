import langid

def detect_language(text: str) -> str:
    text = text.strip()
    letters = [c for c in text if c.isalpha()]

    if not letters:
        return "other"

    # Если все буквы латиница → английский
    if all(ord(c) < 128 for c in letters):
        return "en"

    # Если все буквы кириллица → русский/украинский
    if all('\u0400' <= c <= '\u04FF' for c in letters):
        return "ru"

    # Иначе стандартная классификация через langid
    lang, _ = langid.classify(text)
    if lang in ("ru", "uk"):
        return "ru"
    elif lang == "en":
        return "en"
    else:
        return "other"


def label_to_numeric(label: str) -> float:
    lab = (label or "").lower().strip()
    if not lab:
        return 0.5

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