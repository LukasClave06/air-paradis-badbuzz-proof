import re


def basic_clean(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"@\\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()

    return text