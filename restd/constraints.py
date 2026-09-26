"""Compile query constraints against product metadata."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[./-][^\W_]+)*", re.UNICODE)
NUMBER_PATTERN = re.compile(r"\d+(?:[./-]\d+)*")
MEASUREMENT_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*"
    r"(millimeters?|millimetres?|mm|centimeters?|centimetres?|cm|"
    r"meters?|metres?|m|inches?|inch|in|pulgadas?|インチ|"
    r"milliliters?|millilitres?|ml|liters?|litres?|l|litros?|リットル|"
    r"milligrams?|mg|grams?|gramos?|g|kilograms?|kilogramos?|kg)"
    r"(?![\w])",
    re.IGNORECASE,
)
CJK_RUN_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]+")
COLOR_ALIASES = {
    "black": ("black", "negro", "negra", "黒", "ブラック"),
    "white": ("white", "blanco", "blanca", "白", "ホワイト"),
    "red": ("red", "rojo", "roja", "赤", "レッド"),
    "blue": ("blue", "azul", "青", "ブルー"),
    "green": ("green", "verde", "緑", "グリーン"),
    "gray": ("gray", "grey", "gris", "グレー", "灰色"),
    "silver": ("silver", "plateado", "plateada", "銀", "シルバー"),
    "gold": ("gold", "dorado", "dorada", "金", "ゴールド"),
    "pink": ("pink", "rosa", "ピンク"),
    "purple": ("purple", "morado", "morada", "violeta", "紫", "パープル"),
    "brown": ("brown", "marrón", "marron", "café", "cafe", "茶", "ブラウン"),
    "beige": ("beige", "ベージュ"),
    "yellow": ("yellow", "amarillo", "amarilla", "黄", "イエロー"),
    "orange": ("orange", "naranja", "オレンジ"),
    "clear": ("clear", "transparente", "透明", "クリア"),
    "navy": ("navy", "marino", "紺", "ネイビー"),
    "teal": ("teal", "turquesa", "ティール"),
    "maroon": ("maroon", "granate", "えんじ", "マルーン"),
    "multicolor": ("multicolor", "multicolour", "multicolorido", "マルチカラー"),
}
NEGATORS = frozenset({"without", "no", "not", "sin"})
NEGATION_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "to",
        "of",
        "in",
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "y",
        "o",
        "para",
        "con",
        "de",
    }
)

MEASUREMENT_UNITS = {
    "millimeter": ("mm", 1.0),
    "millimeters": ("mm", 1.0),
    "millimetre": ("mm", 1.0),
    "millimetres": ("mm", 1.0),
    "mm": ("mm", 1.0),
    "centimeter": ("mm", 10.0),
    "centimeters": ("mm", 10.0),
    "centimetre": ("mm", 10.0),
    "centimetres": ("mm", 10.0),
    "cm": ("mm", 10.0),
    "meter": ("mm", 1000.0),
    "meters": ("mm", 1000.0),
    "metre": ("mm", 1000.0),
    "metres": ("mm", 1000.0),
    "m": ("mm", 1000.0),
    "inch": ("mm", 25.4),
    "inches": ("mm", 25.4),
    "in": ("mm", 25.4),
    "pulgada": ("mm", 25.4),
    "pulgadas": ("mm", 25.4),
    "インチ": ("mm", 25.4),
    "milliliter": ("ml", 1.0),
    "milliliters": ("ml", 1.0),
    "millilitre": ("ml", 1.0),
    "millilitres": ("ml", 1.0),
    "ml": ("ml", 1.0),
    "liter": ("ml", 1000.0),
    "liters": ("ml", 1000.0),
    "litre": ("ml", 1000.0),
    "litres": ("ml", 1000.0),
    "litro": ("ml", 1000.0),
    "litros": ("ml", 1000.0),
    "l": ("ml", 1000.0),
    "リットル": ("ml", 1000.0),
    "milligram": ("mg", 1.0),
    "milligrams": ("mg", 1.0),
    "mg": ("mg", 1.0),
    "gram": ("mg", 1000.0),
    "grams": ("mg", 1000.0),
    "gramo": ("mg", 1000.0),
    "gramos": ("mg", 1000.0),
    "g": ("mg", 1000.0),
    "kilogram": ("mg", 1_000_000.0),
    "kilograms": ("mg", 1_000_000.0),
    "kilogramo": ("mg", 1_000_000.0),
    "kilogramos": ("mg", 1_000_000.0),
    "kg": ("mg", 1_000_000.0),
}


def _text(value, *, unicode_nfkc: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(str(item) for item in value.values())
    value = str(value)
    if unicode_nfkc:
        value = unicodedata.normalize("NFKC", value)
    return value.casefold()


def normalize_text(value, *, unicode_nfkc: bool = False) -> str:
    return " ".join(TOKEN_PATTERN.findall(_text(value, unicode_nfkc=unicode_nfkc)))


def tokenize(value, *, unicode_nfkc: bool = False) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(_text(value, unicode_nfkc=unicode_nfkc)))


def canonical_measurements(value, *, unicode_nfkc: bool = False) -> tuple[str, ...]:
    """Return unit-normalized measurements such as ``0.5 L -> 500ml``."""

    raw = _text(value, unicode_nfkc=unicode_nfkc)
    result = set()
    for number, raw_unit in MEASUREMENT_PATTERN.findall(raw):
        unit = raw_unit.casefold()
        canonical = MEASUREMENT_UNITS.get(unit)
        if canonical is None:
            continue
        base_unit, factor = canonical
        numeric = float(number.replace(",", ".")) * factor
        rounded = round(numeric, 6)
        if abs(rounded - round(rounded)) <= 1e-6:
            formatted = str(int(round(rounded)))
        else:
            formatted = (f"{rounded:.6f}").rstrip("0").rstrip(".")
        result.add(f"{formatted}{base_unit}")
    return tuple(sorted(result))


def _contains_lexeme(normalized_text: str, lexeme: str) -> bool:
    """Match word-like Latin phrases exactly and CJK phrases by substring."""
    if not lexeme:
        return False
    if any(ord(character) > 127 for character in lexeme):
        return lexeme in normalized_text
    return f" {lexeme} " in f" {normalized_text} "


def _cjk_ngrams(value: str, *, unicode_nfkc: bool = False) -> set[str]:
    result = set()
    normalized = _text(value, unicode_nfkc=unicode_nfkc)
    for run in CJK_RUN_PATTERN.findall(normalized):
        for width in (2, 3):
            result.update(
                run[index : index + width]
                for index in range(max(0, len(run) - width + 1))
            )
    return result


@dataclass(frozen=True)
class QueryConstraints:
    colors: tuple[str, ...]
    numbers: tuple[str, ...]
    measurements: tuple[str, ...]
    brands: tuple[str, ...]
    excluded: tuple[str, ...]
    lexical: tuple[str, ...] = ()

    @property
    def structured_count(self) -> int:
        return (
            len(self.colors)
            + len(self.numbers)
            + len(self.measurements)
            + len(self.brands)
            + len(self.excluded)
        )

    @property
    def count(self) -> int:
        return self.structured_count + len(self.lexical)

    @property
    def types(self) -> tuple[str, ...]:
        result = []
        for name in (
            "colors",
            "numbers",
            "measurements",
            "brands",
            "excluded",
            "lexical",
        ):
            if getattr(self, name):
                result.append(name)
        return tuple(result)


class LexicalCatalog:
    """Compact inverted index over product metadata."""

    def __init__(
        self,
        metadata_path: str,
        minimum_brand_frequency: int = 2,
        *,
        max_lexical_features: int = 0,
        lexical_weight: float = 0.0,
        max_lexical_df_ratio: float = 0.03,
        unicode_nfkc: bool = False,
        enable_measurements: bool = False,
        enable_negation: bool = True,
    ) -> None:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        item_count = len(metadata)
        if sorted(int(key) for key in metadata) != list(range(item_count)):
            raise ValueError("item metadata ids must be contiguous from zero")
        token_rows = defaultdict(list)
        color_rows = defaultdict(list)
        number_rows = defaultdict(list)
        measurement_rows = defaultdict(list)
        gram_rows = defaultdict(list)
        brand_rows = defaultdict(list)
        brand_counts = Counter()
        raw_brands = []
        for item_index in range(item_count):
            record = metadata[str(item_index)]
            brand = normalize_text(
                record.get("product_brand"), unicode_nfkc=unicode_nfkc
            )
            raw_brands.append(brand)
            if brand:
                brand_counts[brand] += 1
            lexical = " ".join(
                filter(
                    None,
                    (
                        normalize_text(
                            record.get("product_title"),
                            unicode_nfkc=unicode_nfkc,
                        ),
                        brand,
                        normalize_text(
                            record.get("product_color"),
                            unicode_nfkc=unicode_nfkc,
                        ),
                    ),
                )
            )
            for token in set(tokenize(lexical, unicode_nfkc=unicode_nfkc)):
                token_rows[token].append(item_index)
            if max_lexical_features > 0:
                for gram in _cjk_ngrams(lexical, unicode_nfkc=unicode_nfkc):
                    gram_rows[gram].append(item_index)
            for number in set(NUMBER_PATTERN.findall(lexical)):
                number_rows[number].append(item_index)
            if enable_measurements:
                for measurement in canonical_measurements(
                    " ".join(
                        filter(
                            None,
                            (
                                record.get("product_title"),
                                record.get("product_color"),
                            ),
                        )
                    ),
                    unicode_nfkc=unicode_nfkc,
                ):
                    measurement_rows[measurement].append(item_index)
            for canonical, aliases in COLOR_ALIASES.items():
                if any(
                    _contains_lexeme(
                        lexical,
                        normalize_text(alias, unicode_nfkc=unicode_nfkc),
                    )
                    for alias in aliases
                ):
                    color_rows[canonical].append(item_index)
        for item_index, brand in enumerate(raw_brands):
            if brand and brand_counts[brand] >= minimum_brand_frequency:
                brand_rows[brand].append(item_index)
        self.item_count = item_count
        self.token_to_items = {
            token: np.asarray(rows, dtype=np.int32)
            for token, rows in token_rows.items()
        }
        self.number_to_items = {
            number: np.asarray(rows, dtype=np.int32)
            for number, rows in number_rows.items()
        }
        self.measurement_to_items = {
            measurement: np.asarray(rows, dtype=np.int32)
            for measurement, rows in measurement_rows.items()
        }
        self.color_to_items = {
            color: np.asarray(rows, dtype=np.int32)
            for color, rows in color_rows.items()
        }
        self.brand_to_items = {
            brand: np.asarray(rows, dtype=np.int32)
            for brand, rows in brand_rows.items()
        }
        self.brands = tuple(
            sorted(self.brand_to_items, key=lambda value: (-len(value), value))
        )
        self.max_lexical_features = int(max_lexical_features)
        self.lexical_weight = float(lexical_weight)
        self.unicode_nfkc = bool(unicode_nfkc)
        self.enable_measurements = bool(enable_measurements)
        self.enable_negation = bool(enable_negation)
        self.lexical_to_items = {}
        self.lexical_idf = {}
        if self.max_lexical_features > 0:
            maximum_df = max(50, int(max_lexical_df_ratio * item_count))
            for token, rows in token_rows.items():
                if (
                    len(token) < 3
                    or token.isdigit()
                    or CJK_RUN_PATTERN.search(token)
                    or not 2 <= len(rows) <= maximum_df
                ):
                    continue
                key = f"t:{token}"
                self.lexical_to_items[key] = np.asarray(rows, dtype=np.int32)
                self.lexical_idf[key] = math.log((item_count + 1) / (len(rows) + 1))
            for gram, rows in gram_rows.items():
                if not 2 <= len(rows) <= maximum_df:
                    continue
                key = f"g:{gram}"
                self.lexical_to_items[key] = np.asarray(rows, dtype=np.int32)
                self.lexical_idf[key] = math.log((item_count + 1) / (len(rows) + 1))

    def compile(self, query: str) -> QueryConstraints:
        normalized = normalize_text(query, unicode_nfkc=self.unicode_nfkc)
        tokens = tokenize(query, unicode_nfkc=self.unicode_nfkc)
        colors = tuple(
            sorted(
                canonical
                for canonical, aliases in COLOR_ALIASES.items()
                if any(
                    _contains_lexeme(
                        normalized,
                        normalize_text(alias, unicode_nfkc=self.unicode_nfkc),
                    )
                    for alias in aliases
                )
            )
        )
        numbers = tuple(sorted(set(NUMBER_PATTERN.findall(normalized))))
        measurements = (
            canonical_measurements(query, unicode_nfkc=self.unicode_nfkc)
            if self.enable_measurements
            else ()
        )
        excluded = []
        if self.enable_negation:
            for index, token in enumerate(tokens[:-1]):
                if token not in NEGATORS:
                    continue
                candidate = tokens[index + 1]
                if candidate not in NEGATION_STOP and len(candidate) >= 2:
                    excluded.append(candidate)
        brands = []
        for brand in self.brands:
            if _contains_lexeme(normalized, brand):
                brands.append(brand)
                if len(brands) == 2:
                    break
        lexical_candidates = set()
        if self.max_lexical_features > 0:
            lexical_candidates.update(
                f"g:{gram}"
                for gram in _cjk_ngrams(normalized, unicode_nfkc=self.unicode_nfkc)
            )
            lexical_candidates.update(
                f"t:{token}"
                for token in tokens
                if len(token) >= 3
                and not token.isdigit()
                and not CJK_RUN_PATTERN.search(token)
            )
        lexical = tuple(
            sorted(
                (key for key in lexical_candidates if key in self.lexical_idf),
                key=lambda key: (-self.lexical_idf[key], key),
            )[: self.max_lexical_features]
        )
        return QueryConstraints(
            colors=colors,
            numbers=numbers,
            measurements=measurements,
            brands=tuple(brands),
            excluded=tuple(sorted(set(excluded))),
            lexical=lexical,
        )

    def item_scores(
        self, constraints: QueryConstraints, *, lexical_weight: float | None = None
    ) -> np.ndarray:
        scores = np.zeros(self.item_count, dtype=np.float32)
        if constraints.count == 0:
            return scores
        for color in constraints.colors:
            scores -= 0.20
            rows = self.color_to_items.get(color)
            if rows is not None:
                scores[rows] += 1.20
        for number in constraints.numbers:
            scores -= 0.20
            rows = self.number_to_items.get(number)
            if rows is not None:
                scores[rows] += 1.20
        for measurement in constraints.measurements:
            scores -= 0.20
            rows = self.measurement_to_items.get(measurement)
            if rows is not None:
                scores[rows] += 1.20
        for brand in constraints.brands:
            scores -= 0.25
            rows = self.brand_to_items.get(brand)
            if rows is not None:
                scores[rows] += 1.25
        for token in constraints.excluded:
            rows = self.token_to_items.get(token)
            if rows is not None:
                scores[rows] -= 2.0
        if constraints.structured_count:
            scores /= constraints.structured_count
        effective_lexical_weight = (
            self.lexical_weight if lexical_weight is None else float(lexical_weight)
        )
        if constraints.lexical and effective_lexical_weight:
            lexical_scores = np.zeros(self.item_count, dtype=np.float32)
            denominator = 0.0
            for feature in constraints.lexical:
                weight = self.lexical_idf[feature]
                lexical_scores[self.lexical_to_items[feature]] += weight
                denominator += weight
            if denominator:
                scores += effective_lexical_weight * lexical_scores / denominator
        return scores
