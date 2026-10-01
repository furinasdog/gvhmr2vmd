"""Semantic SMPL-to-MMD bone mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoneRule:
    semantic: str
    source_index: int
    candidates: tuple[str, ...]
    required: bool = True


# Ordered from roots to leaves. Japanese names are authoritative for MMD;
# English aliases support models imported from other common toolchains.
BONE_RULES = (
    BoneRule("root", 0, ("センター", "Center", "center", "全ての親", "Root", "root")),
    BoneRule("spine1", 3, ("上半身", "Upper Body", "Spine", "spine")),
    BoneRule("spine2", 6, ("上半身1", "上半身3", "Spine1", "spine_01"), required=False),
    BoneRule("spine3", 9, ("上半身2", "Upper Body 2", "Chest", "chest")),
    BoneRule("neck", 12, ("首", "Neck", "neck")),
    BoneRule("head", 15, ("頭", "Head", "head")),
    BoneRule("left_collar", 13, ("左肩P", "肩P.L", "左肩", "肩.L", "shoulderP_L", "LeftShoulder"), required=False),
    BoneRule("left_shoulder", 16, ("左腕", "腕.L", "UpperArm_L", "upper_arm.L", "LeftArm")),
    BoneRule("left_elbow", 18, ("左ひじ", "ひじ.L", "左肘", "Elbow_L", "forearm.L", "LeftForeArm")),
    BoneRule("left_wrist", 20, ("左手首", "手首.L", "Wrist_L", "hand.L", "LeftHand")),
    BoneRule("right_collar", 14, ("右肩P", "肩P.R", "右肩", "肩.R", "shoulderP_R", "RightShoulder"), required=False),
    BoneRule("right_shoulder", 17, ("右腕", "腕.R", "UpperArm_R", "upper_arm.R", "RightArm")),
    BoneRule("right_elbow", 19, ("右ひじ", "ひじ.R", "右肘", "Elbow_R", "forearm.R", "RightForeArm")),
    BoneRule("right_wrist", 21, ("右手首", "手首.R", "Wrist_R", "hand.R", "RightHand")),
    BoneRule("left_hip", 1, ("左足", "足.L", "Leg_L", "thigh.L", "LeftUpLeg")),
    BoneRule("left_knee", 4, ("左ひざ", "ひざ.L", "左膝", "Knee_L", "shin.L", "LeftLeg")),
    BoneRule("left_ankle", 7, ("左足首", "足首.L", "Ankle_L", "foot.L", "LeftFoot")),
    BoneRule("left_foot", 10, ("左つま先", "つま先.L", "左足先EX", "足先EX.L", "Toe_L", "toe.L", "LeftToeBase"), required=False),
    BoneRule("right_hip", 2, ("右足", "足.R", "Leg_R", "thigh.R", "RightUpLeg")),
    BoneRule("right_knee", 5, ("右ひざ", "ひざ.R", "右膝", "Knee_R", "shin.R", "RightLeg")),
    BoneRule("right_ankle", 8, ("右足首", "足首.R", "Ankle_R", "foot.R", "RightFoot")),
    BoneRule("right_foot", 11, ("右つま先", "つま先.R", "右足先EX", "足先EX.R", "Toe_R", "toe.R", "RightToeBase"), required=False),
)

RULE_BY_SEMANTIC = {rule.semantic: rule for rule in BONE_RULES}


@dataclass(frozen=True)
class HandBoneRule:
    semantic: str
    side: str
    parent_semantic: str | None
    landmark_pair: tuple[int, int]
    candidates: tuple[str, ...]


def _hand_rules(side: str, prefix: str, suffix: str):
    jp = "左" if side == "left" else "右"
    lr = "L" if side == "left" else "R"
    fingers = (
        ("thumb", "親指", (1, 2, 3, 4), ("０", "１", "２")),
        ("index", "人指", (5, 6, 7, 8), ("１", "２", "３")),
        ("middle", "中指", (9, 10, 11, 12), ("１", "２", "３")),
        ("ring", "薬指", (13, 14, 15, 16), ("１", "２", "３")),
        ("little", "小指", (17, 18, 19, 20), ("１", "２", "３")),
    )
    rules = []
    for english, japanese, landmarks, numbers in fingers:
        parent = None
        for index, number in enumerate(numbers):
            semantic = f"{side}_{english}{index + 1}"
            ascii_number = str(index if english == "thumb" else index + 1)
            candidates = (
                f"{jp}{japanese}{number}",
                f"{japanese}{number}.{lr}",
                f"{japanese}{ascii_number}.{lr}",
                f"{english}.{index + 1:02d}.{lr}",
                f"{prefix}{english}{index + 1}{suffix}",
            )
            rules.append(
                HandBoneRule(
                    semantic,
                    side,
                    parent,
                    (landmarks[index], landmarks[index + 1]),
                    candidates,
                )
            )
            parent = semantic
    return tuple(rules)


HAND_BONE_RULES = _hand_rules("left", "", "_L") + _hand_rules("right", "", "_R")


def resolve_bone_map(bone_names) -> tuple[dict[str, str], list[str]]:
    """Resolve rules against an iterable of actual armature bone names."""
    names = list(bone_names)
    exact = set(names)
    folded = {name.casefold(): name for name in names}
    used: set[str] = set()
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for rule in BONE_RULES:
        match = None
        for candidate in rule.candidates:
            if candidate in exact and candidate not in used:
                match = candidate
                break
            actual = folded.get(candidate.casefold())
            if actual is not None and actual not in used:
                match = actual
                break
        if match is None:
            if rule.required:
                missing.append(f"{rule.semantic} ({'/'.join(rule.candidates[:2])})")
            continue
        resolved[rule.semantic] = match
        used.add(match)
    return resolved, missing


def resolve_hand_bone_map(bone_names) -> tuple[dict[str, str], list[str]]:
    names = list(bone_names)
    exact = set(names)
    folded = {name.casefold(): name for name in names}
    resolved = {}
    missing = []
    for rule in HAND_BONE_RULES:
        match = next(
            (
                candidate if candidate in exact else folded.get(candidate.casefold())
                for candidate in rule.candidates
                if candidate in exact or candidate.casefold() in folded
            ),
            None,
        )
        if match is None:
            missing.append(rule.semantic)
        else:
            resolved[rule.semantic] = match
    return resolved, missing
