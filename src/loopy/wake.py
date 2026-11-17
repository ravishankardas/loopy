"""Wake word: "Hey Loopy" (also "Loopy", "Okay Loopy", ...).

    split_wake("Hey Loopy, open Chrome.")  ->  (True, "open chrome")
    split_wake("open chrome")              ->  (False, "open chrome")

Whisper doesn't know the name "Loopy", so it may write it a few different
ways; all of those count. The name must come first (after an optional
greeting), so "I said loopy earlier" doesn't wake it.
"""

import re

NAME_SPELLINGS = {
    "loopy", "loopie", "loopey", "loopee", "loopi", "luppy", "lupi", "lupy",
    "lupie", "lupe", "loupy", "loupie", "loupee",
}
GREETINGS = {"hey", "hi", "hello", "ok", "okay", "yo", "a", "hay"}


def split_wake(text: str) -> tuple[bool, str]:
    """Return (wake word said?, the rest of the sentence, lowercased)."""
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    i = 0
    while i < len(words) and i < 2 and words[i] in GREETINGS:
        i += 1
    if i < len(words) and words[i] in NAME_SPELLINGS:
        return True, " ".join(words[i + 1:])
    # Sometimes it comes out as two words: "Lou pee", "loo py".
    if i + 1 < len(words) and words[i] + words[i + 1] in NAME_SPELLINGS:
        return True, " ".join(words[i + 2:])
    return False, " ".join(words)
