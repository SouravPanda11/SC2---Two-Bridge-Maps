"""Canonical Two-Bridge map metadata used by the scripted evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapVariant:
    """Static metadata for one benchmark map."""

    name: str
    enemy_count: int

    @property
    def filename(self) -> str:
        return f"TwoBridgeMap_{self.name}.SC2Map"

    @property
    def registry_name(self) -> str:
        # A private name avoids clashes with environment modules imported in the
        # same interpreter. PySC2 uses the class name as the registry key.
        return f"ScriptedTwoBridgeMap{self.name.replace('_', '')}"


MAP_VARIANTS: dict[str, MapVariant] = {
    name: MapVariant(name=name, enemy_count=enemy_count)
    for enemy_count, version in ((3, "V1"), (5, "V2"), (8, "V3"))
    for name in (
        f"{version}_Base",
        f"{version}_Combat",
        f"{version}_Navigate",
    )
}

CANONICAL_VARIANT_NAMES: tuple[str, ...] = tuple(MAP_VARIANTS)


def get_variant(name: str) -> MapVariant:
    try:
        return MAP_VARIANTS[name]
    except KeyError as exc:
        choices = ", ".join(CANONICAL_VARIANT_NAMES)
        raise ValueError(f"Unknown map variant {name!r}. Choose one of: {choices}") from exc
