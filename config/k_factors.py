"""Tournament importance weights used by the Elo rating engine."""

from __future__ import annotations

import unicodedata

DEFAULT_K_FACTOR = 20

K_FACTORS = {
    "Friendly": 20,
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "Copa America": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Oceania Nations Cup": 50,

    "UEFA Euro qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "Gold Cup qualification": 40,

    "UEFA Nations League": 35,
    "CONCACAF Nations League": 35,

    "AFF Championship": 30,
    "ASEAN Championship": 30,
    "Gulf Cup": 30,
    "SAFF Cup": 30,
    "EAFF Championship": 30,
    "WAFF Championship": 30,
    "UNCAF Cup": 30,
    "CECAFA Cup": 30,
    "COSAFA Cup": 30,
    "CFU Caribbean Cup": 30,
    "Arab Cup": 30,
}


def _normalize_tournament_name(tournament: str) -> str:
    normalized = unicodedata.normalize("NFKD", tournament.strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.split())


def get_k_factor(tournament: str | None, default: int = DEFAULT_K_FACTOR) -> int:
    """Return the Elo K-factor for a tournament, falling back to friendlies."""
    if tournament is None:
        return default

    tournament_name = str(tournament).strip()
    if tournament_name in K_FACTORS:
        return K_FACTORS[tournament_name]

    return K_FACTORS.get(_normalize_tournament_name(tournament_name), default)
