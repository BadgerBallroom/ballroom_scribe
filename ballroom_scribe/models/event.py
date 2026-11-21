from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Represents raw ballroom event details extracted from an emcee announcement.

    Every music valley produces at least one `Event`. If no speech was transcribed in the music valley, or the LLM did
    not detect an announcement in the transcript, a placeholder event with empty text fields marks the music valley
    instead.

    Attributes:
        event_number: The event number in the competition program, if announced.
        flight: The flight number or ordinal within the event, if announced.
        heat: The heat number or ordinal within the round, if announced.
        age_division: The age division, such as youth, adult, senior, or all ages.
        category: The competitor category, such as amateur, professional, pro-am, or rookie/vet.
        level: The ballroom proficiency level, such as bronze, silver, gold, novice, etc.
        style: The category of the dance, such as smooth, standard, rhythm, or Latin.
        dance: The dance itself, such as waltz, quickstep, rumba, or jive.
        dance_ordinal: Which dance in a multi-dance sequence is being announced, such as first or second.
        specialness: Non-competitive event designation, such as exhibition, critique, fun, or showcase.
        round: The round, with the last rounds being quarterfinal, semifinal, and final.
        announcement_quote: What the emcee spoke to announce the event.
        announcement_start_seconds:
            The number of seconds since the beginning of the original audio file that the emcee began to announce the
            event. For a placeholder event, this is the midpoint of the music valley instead.
        speech_detected: Whether any words were transcribed in the music valley.
        event_detected: Whether the LLM detected an event announcement in the transcribed words.
    """

    event_number: str = ""
    flight: str = ""
    heat: str = ""
    age_division: str = ""
    category: str = ""
    level: str = ""
    style: str = ""
    dance: str = ""
    dance_ordinal: str = ""
    specialness: str = ""
    round: str = ""
    announcement_quote: str = ""
    announcement_start_seconds: float
    speech_detected: bool
    event_detected: bool
