from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """Represents raw ballroom event details extracted from an emcee announcement.

    Attributes:
        level: The ballroom proficiency level, such as bronze, silver, gold, novice, etc.
        style: The category of the dance, such as smooth, standard, rhythm, or Latin.
        dance: The dance itself, such as waltz, quickstep, rumba, or jive.
        round: The round, with the last rounds being quarterfinal, semifinal, and final.
        announcement_quote: What the emcee spoke to announce the event.
        announcement_start_seconds:
            The number of seconds since the beginning of the original audio file that the emcee began to announce the
            event.
    """

    level: str
    style: str
    dance: str
    round: str
    announcement_quote: str
    announcement_start_seconds: float
