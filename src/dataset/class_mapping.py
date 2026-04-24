"""
class_mapping.py
----------------
Handles the mapping between raw YOLO class IDs and the target
classification labels used by the processed dataset.

WHY THIS MODULE EXISTS
----------------------
YOLO detection datasets can use class IDs in many ways:

  Case A — Single class (class 0 = "person")
      The dataset treats everyone as "person" and makes no
      distinction between normal / drowning / struggling.
      → You CANNOT derive drowning labels from bounding boxes alone.
        Manual relabelling or a different source is required.

  Case B — Multiple classes matching target labels
      class 0 = "normal", class 1 = "drowning", class 2 = "struggling"
      → Direct mapping is possible and clean.

  Case C — Multiple classes with different names / ordering
      class 0 = "swim", class 1 = "drown", class 2 = "distress"
      → Remapping table is needed.

  Case D — Mixed / noisy labels
      → Some classes may need to be merged, renamed, or ignored.

HOW TO USE
----------
1. Run dataset_analyzer.py first to discover what class IDs exist.
2. Edit CLASS_MAPPING below to match your dataset's actual class IDs.
3. Any class ID NOT present in CLASS_MAPPING will be skipped during
   cropping (logged as a warning, never crashes).

The target label names used by the processed dataset are:
    "normal", "drowning", "struggling"

If your dataset has only class 0 = "person", set SINGLE_CLASS_MODE = True
and the class mapping becomes irrelevant — you will need to label manually.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CONFIGURATION — edit this section after running dataset_analyzer.py
# ---------------------------------------------------------------------------

# Set to True if your raw dataset has only one class (e.g. "person") and
# does NOT distinguish between normal/drowning/struggling.
# In that case the pipeline will save all crops under "unknown" and you
# will need to relabel them manually or obtain a labelled dataset.
SINGLE_CLASS_MODE: bool = False

# Mapping: raw YOLO class_id → target label string.
# Edit these entries to match what data.yaml says.
#
# Example (Case B — direct match):
#   CLASS_MAPPING = {0: "normal", 1: "drowning", 2: "struggling"}
#
# Example (Case C — remapping):
#   CLASS_MAPPING = {0: "normal", 1: "drowning", 2: "drowning", 3: "struggling"}
#
# Example (Case A — single class, all persons):
#   SINGLE_CLASS_MODE = True
#   CLASS_MAPPING = {0: "unknown"}   # will be relabelled later
#
CLASS_MAPPING: dict[int, str] = {
    0: "drowning",
    1: "normal",       # Swimming = normal behaviour
    2: "struggling",   # out of water = struggling / distress
}

# Target label set (used for output folder creation and validation).
TARGET_LABELS: list[str] = ["normal", "drowning", "struggling"]

# ---------------------------------------------------------------------------
# Runtime helper (do not edit below unless you know what you are doing)
# ---------------------------------------------------------------------------

def map_class_id(class_id: int) -> str | None:
    """
    Convert a raw YOLO class_id to a target label string.

    Returns
    -------
    The target label string, or None if this class_id should be skipped.
    """
    if SINGLE_CLASS_MODE:
        return "unknown"
    return CLASS_MAPPING.get(class_id, None)


def get_active_labels() -> list[str]:
    """Return the list of target labels that are actually reachable."""
    if SINGLE_CLASS_MODE:
        return ["unknown"]
    return list(set(CLASS_MAPPING.values()))


def print_mapping_summary() -> None:
    """Print the current class mapping configuration."""
    print("\n  CLASS MAPPING CONFIGURATION")
    print(f"  Single-class mode : {SINGLE_CLASS_MODE}")
    if SINGLE_CLASS_MODE:
        print("  All crops → 'unknown' (manual relabelling required)")
    else:
        for cid, label in sorted(CLASS_MAPPING.items()):
            print(f"    class {cid} → '{label}'")
    unmapped_warning = [
        cid for cid in range(10)
        if cid not in CLASS_MAPPING and not SINGLE_CLASS_MODE
    ]
    if unmapped_warning[:3]:
        print(f"  [NOTE] Class IDs not in mapping will be SKIPPED during cropping.")
