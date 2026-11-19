"""Detects whether a transcribed word ends a sentence."""

_SENTENCE_ENDINGS = (".", "!", "?")
# Quotes and brackets that may close after the punctuation that ends a sentence, as in `floor."` or `(again!)`.
_CLOSING_PUNCTUATION = "\"')]}\u201d\u2019"


def ends_sentence(text: str) -> bool:
    """Returns whether `text` ends with a period, an exclamation mark, or a question mark.

    Closing quotes and brackets after the punctuation are ignored, so `floor."` ends a sentence. Abbreviations such as
    `Mr.` count as sentence ends too.

    Args:
        text: A transcribed word, including any punctuation attached to it.
    """
    return text.rstrip(_CLOSING_PUNCTUATION).endswith(_SENTENCE_ENDINGS)
