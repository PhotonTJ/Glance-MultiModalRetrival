from __future__ import annotations

import re
from dataclasses import dataclass, field


COLORS = {
    "black", "white", "gray", "grey", "red", "orange", "yellow", "green",
    "blue", "navy", "purple", "pink", "brown", "beige", "mustard",
}
GARMENTS = {
    "bag", "belt", "blazer", "blouse", "cape", "cardigan", "coat", "dress",
    "glasses", "glove", "hat", "hoodie", "jacket", "jumpsuit", "pants",
    "raincoat", "scarf", "shirt", "shoe", "shorts", "skirt", "sock", "suit",
    "sweater", "t-shirt", "tee", "tie", "tights", "top", "trousers", "vest", "watch",
}
ENVIRONMENTS = {
    "office": {"office", "workplace", "boardroom"},
    "urban_street": {"street", "city", "urban", "sidewalk"},
    "park": {"park", "garden"},
    "home": {"home", "living room", "bedroom"},
}
STYLES = {
    "professional": {"professional", "business", "formal"},
    "casual": {"casual", "weekend", "relaxed"},
    "outerwear": {"outerwear"},
}
ACTIVITIES = {"sitting", "walking", "standing", "working", "running"}
OBJECTS = {"bench", "desk", "chair", "laptop", "umbrella"}
COLOR_NORMALIZATION = {"grey": "gray", "mustard": "yellow"}
# Keep query terms in the same family as Fashionpedia labels.  In particular,
# mapping ``pants`` to ``trousers`` made a real ``pants`` annotation fail an
# otherwise exact garment--color comparison.
GARMENT_NORMALIZATION = {
    "tee": "t-shirt", "tees": "t-shirt", "t-shirts": "t-shirt",
    "shoes": "shoe", "shirts": "shirt", "dresses": "dress", "ties": "tie",
    "socks": "sock", "skirts": "skirt", "blouses": "blouse", "watches": "watch",
    "belts": "belt", "bags": "bag", "gloves": "glove", "hats": "hat",
    "hoodies": "hoodie", "jackets": "jacket", "coats": "coat", "suits": "suit",
    "sweaters": "sweater", "vests": "vest", "scarves": "scarf", "raincoats": "raincoat",
    "cardigans": "cardigan", "capes": "cape", "jumpsuits": "jumpsuit",
}


def normalize_garment(value: str) -> str:
    return GARMENT_NORMALIZATION.get(value.strip().lower(), value.strip().lower())


def garment_matches(requested: str, label: str) -> bool:
    """Match query vocabulary to Fashionpedia's compound category labels."""
    requested = normalize_garment(requested)
    parts = [normalize_garment(part.strip().replace("_", " ")) for part in label.lower().split("_/_")]
    return requested in parts


@dataclass(frozen=True)
class QueryConstraints:
    bindings: tuple[tuple[str, str], ...] = ()
    garments: frozenset[str] = frozenset()
    colors: frozenset[str] = frozenset()
    environment: str | None = None
    styles: frozenset[str] = frozenset()
    activity: str | None = None
    objects: frozenset[str] = frozenset()


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text) is not None


def parse_query(text: str) -> QueryConstraints:
    normalized = text.lower().replace("bright ", "")
    bindings: list[tuple[str, str]] = []
    # Local binding is intentional: it differentiates red tie + white shirt from the swap.
    garment_forms = sorted(GARMENTS | set(GARMENT_NORMALIZATION), key=len, reverse=True)
    pattern = rf"\b({'|'.join(sorted(COLORS, key=len, reverse=True))})\s+(?:\w+\s+)?({'|'.join(map(re.escape, garment_forms))})\b"
    for color, garment in re.findall(pattern, normalized):
        color = COLOR_NORMALIZATION.get(color, color)
        garment = normalize_garment(garment)
        bindings.append((garment, color))

    garments = {normalize_garment(g) for g in garment_forms if _contains(normalized, g)}
    colors = {COLOR_NORMALIZATION.get(c, c) for c in COLORS if _contains(normalized, c)}
    environment = next((label for label, terms in ENVIRONMENTS.items() if any(_contains(normalized, t) for t in terms)), None)
    styles = {label for label, terms in STYLES.items() if any(_contains(normalized, t) for t in terms)}
    activity = next((a for a in ACTIVITIES if _contains(normalized, a)), None)
    objects = {o for o in OBJECTS if _contains(normalized, o)}
    return QueryConstraints(tuple(bindings), frozenset(garments), frozenset(colors), environment, frozenset(styles), activity, frozenset(objects))
