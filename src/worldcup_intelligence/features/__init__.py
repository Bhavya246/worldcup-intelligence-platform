"""Feature engineering utilities for match prediction models."""

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "build_feature_dataset",
    "build_feature_rows",
    "result_label",
    "write_feature_dataset",
]


def __getattr__(name: str):
    if name in __all__:
        from worldcup_intelligence.features import builder

        return getattr(builder, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
