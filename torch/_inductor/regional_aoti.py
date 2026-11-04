# mypy: allow-untyped-defs
"""Experimental regional AOTInductor marking and discovery API."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import torch.nn as nn


_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

__all__ = [
    "AOTIRegionDiscoveryError",
    "AOTIRegionError",
    "RegionConfig",
    "RegionResult",
    "mark_region",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AOTIRegionError(RuntimeError):
    """Base class for all regional AOTI errors."""

    def __init__(self, message: str, result=None) -> None:
        super().__init__(message)
        self.result = result


class AOTIRegionDiscoveryError(AOTIRegionError):
    """Raised when selected regions cannot be represented safely."""


# ---------------------------------------------------------------------------
# Config / info dataclasses
# ---------------------------------------------------------------------------
class _UnsetType(Enum):
    UNSET = "unset"

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _UnsetType.UNSET


@dataclasses.dataclass(init=False)
class RegionConfig:
    """Artifact-related configuration for one selected region.

    Attributes:
        dynamic_shapes: Dynamic-shape specs keyed by ``forward`` argument name,
            as in the dict form of ``torch.export.export(dynamic_shapes=...)``.
        inductor_configs: Inductor config overrides passed to
            ``aoti_compile_and_package``.
        rtol: Relative tolerance for the compiled-versus-eager accuracy check.
        atol: Absolute tolerance for the compiled-versus-eager accuracy check.
    """

    dynamic_shapes: dict[str, Any] | None | _UnsetType
    inductor_configs: dict[str, Any] | None | _UnsetType
    rtol: float | None | _UnsetType
    atol: float | None | _UnsetType

    def __init__(
        self,
        *,
        dynamic_shapes=_UNSET,
        inductor_configs=_UNSET,
        rtol=_UNSET,
        atol=_UNSET,
    ) -> None:
        if not (
            dynamic_shapes is _UNSET
            or dynamic_shapes is None
            or isinstance(dynamic_shapes, dict)
        ):
            raise TypeError(
                "RegionConfig.dynamic_shapes must be a dict keyed by forward "
                f"argument name, got {type(dynamic_shapes).__name__}"
            )
        self.dynamic_shapes = dynamic_shapes
        self.inductor_configs = inductor_configs
        self.rtol = rtol
        self.atol = atol


@dataclasses.dataclass
class RegionResult:
    """One marked region and everything recorded about it.

    Attributes:
        name: Path of the module in the model (``""`` for the root).
        module: The marked ``nn.Module`` instance.
        config: Effective ``RegionConfig`` (class mark, instance mark, defaults).
        aliases: Every path at which the same module instance appears.
    """

    name: str
    module: nn.Module = dataclasses.field(repr=False)
    config: RegionConfig = dataclasses.field(repr=False)
    aliases: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Marking / discovery
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS = {
    "dynamic_shapes": None,
    "inductor_configs": None,
    "rtol": 1e-3,
    "atol": 1e-3,
}


def _validate_config(config: RegionConfig | None) -> RegionConfig:
    if config is None:
        return RegionConfig()
    if not isinstance(config, RegionConfig):
        raise TypeError(f"config must be RegionConfig, got {type(config)!r}")
    return config


_MISSING = object()


def mark_region(module=_MISSING, /, *, config: RegionConfig | None = None):
    """Mark one module instance, or one module class, as a region.

    A class mark also applies to subclasses that inherit its ``forward``,
    including the classes swapped in by ``fully_shard``, ``replicate`` and
    parametrizations. A subclass that defines its own ``forward`` (even one
    that only calls ``super().forward()``) is a region only if it is marked.

    A marked module nested inside another marked module is not a separate
    region: the outermost mark wins, so marking the root compiles the whole
    model as one region.
    """
    config = _validate_config(config)

    def decorate(obj):
        if isinstance(obj, type):
            if not issubclass(obj, nn.Module):
                raise TypeError("mark_region can decorate only nn.Module classes")
            setattr(obj, _REGION_CONFIG_ATTR, config)
            return obj
        if not isinstance(obj, nn.Module):
            raise TypeError(f"mark_region expects an nn.Module, got {type(obj)!r}")
        # Instance marks bypass nn.Module.__setattr__ and never touch the class.
        obj.__dict__[_REGION_CONFIG_ATTR] = config
        return obj

    return decorate if module is _MISSING else decorate(module)


def _merge_configs(
    class_config: RegionConfig | None, instance_config: RegionConfig | None
) -> RegionConfig:
    values = {}
    for field, default in _CONFIG_DEFAULTS.items():
        value = getattr(instance_config, field, _UNSET)
        if value is _UNSET:
            value = getattr(class_config, field, _UNSET)
        values[field] = default if value is _UNSET else value
    return RegionConfig(**values)


def _class_region_config(cls: type) -> RegionConfig | None:
    """Class mark that applies to instances of ``cls``.

    Walks the MRO up to the first class that defines ``forward``. Class swaps
    by ``fully_shard``, ``replicate`` and parametrizations add no ``forward``,
    so they resolve to the original class.
    """
    for klass in cls.__mro__:
        config = klass.__dict__.get(_REGION_CONFIG_ATTR)
        if isinstance(config, RegionConfig):
            return config
        if "forward" in klass.__dict__:
            return None
    return None


def _get_region_config(module: nn.Module) -> RegionConfig | None:
    instance_config = module.__dict__.get(_REGION_CONFIG_ATTR)
    class_config = _class_region_config(type(module))
    if not isinstance(instance_config, RegionConfig):
        instance_config = None
    if instance_config is None and class_config is None:
        return None
    return _merge_configs(class_config, instance_config)


def _is_nested_under(name: str, ancestors: list[str]) -> bool:
    """True if ``name`` is a strict descendant of any name in ``ancestors``."""
    for other in ancestors:
        if other == name:
            continue
        if other == "":
            # Root is an ancestor of every non-root module.
            return True
        if name.startswith(other + "."):
            return True
    return False


def _find_marked_regions(model: nn.Module) -> list[RegionResult]:
    """Walk ``model.named_modules()`` and return only top-level marked regions.

    Nested marked regions are skipped in favor of their marked ancestor. A
    marked module that is both nested in another region and a top-level region
    through an alias is rejected. Sharing between regions is not otherwise
    tracked: a submodule, marked or not, that is reachable from two regions is
    part of both.
    """
    marked: dict[int, tuple[nn.Module, RegionConfig, list[str]]] = {}
    for name, mod in model.named_modules(remove_duplicate=False):
        cfg = _get_region_config(mod)
        if cfg is None:
            continue
        if id(mod) not in marked:
            marked[id(mod)] = (mod, cfg, [])
        marked[id(mod)][2].append(name)

    all_names = [name for _, _, aliases in marked.values() for name in aliases]
    regions: list[RegionResult] = []
    for mod, cfg, aliases in marked.values():
        others = [name for name in all_names if name not in aliases]
        nested = [_is_nested_under(name, others) for name in aliases]
        # Nested in a marked ancestor and also a top-level region through
        # another alias: reject instead of picking a side.
        if any(nested) and not all(nested):
            paths = ", ".join(repr(name or "__root__") for name in aliases)
            raise AOTIRegionDiscoveryError(
                f"Aliased selected module crosses a selected-region boundary: {paths}"
            )
        if all(nested):
            continue
        regions.append(
            RegionResult(name=aliases[0], module=mod, config=cfg, aliases=aliases)
        )
    return regions
