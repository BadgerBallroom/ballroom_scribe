from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    """Represents a single word with timestamp metadata from speech-to-text.

    Attributes:
        word: The word itself.
        start: The number of seconds since the beginning of the original audio file that the word started to be spoken.
        end: The number of seconds since the beginning of the original audio file that the word stopped being spoken.
        probability: The likelihood that this word is correct.
    """

    word: str
    start: float
    end: float
    probability: float = 1.0
