"""Humanoid naming aliases shared by conversion and original-rig retargeting."""
import re


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", name.lower().removeprefix("mixamorig:"))


def semantic_mapping(source_names, target_names):
    sources = {normalized(name): name for name in source_names}
    targets = set(target_names)
    result = {name: name for name in source_names if name in targets}
    aliases = {
        "pelvis": ("Hips", "pelvis"), "spine_0": ("Spine", "spine0"),
        "spine_1": ("Chest", "spine1"), "spine_3": ("ChestUp", "UpperChest", "spine3"),
        "neck_0": ("Neck", "neck0"), "head": ("Head",), "jaw": ("Jaw",),
    }
    for side, word in (("L", "Left"), ("R", "Right")):
        for target, candidates in {
            "clavicle": (f"{word} shoulder", f"{word}Shoulder", f"shoulder.{side}"),
            "arm_upper": (f"{word} arm", f"{word}UpperArm", f"upper_arm.{side}"),
            "arm_lower": (f"{word} elbow", f"{word}ForeArm", f"{word}LowerArm", f"forearm.{side}"),
            "hand": (f"{word} wrist", f"{word}Hand", f"hand.{side}"),
            "leg_upper": (f"{word}UpLeg", f"{word}UpperLeg", f"{word} leg", f"thigh.{side}"),
            "leg_lower": (f"{word} knee", f"{word}Leg", f"{word}LowerLeg", f"shin.{side}"),
            "ankle": (f"{word} ankle", f"{word}Foot", f"foot.{side}"),
            "ball": (f"{word} toe", f"{word}ToeBase", f"toe.{side}"),
        }.items():
            aliases[f"{target}_{side}"] = candidates
        eye = next((name for name in (f"eye_0_{side}", f"eye_{side}") if name in targets), None)
        if eye:
            aliases[eye] = (f"Eye_{side}", f"{word}Eye")
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for index in range(3):
                aliases[f"finger_{finger}_{index}_{side}"] = (
                    f"{finger.title()}Finger{index + 1}_{side}", f"{word}Hand{finger.title()}{index + 1}",
                    f'{"thumb" if finger == "thumb" else "f_" + finger}.{index + 1:02d}.{side}',
                )
        for index in range(3):
            aliases[f"ear_{index}_{side}"] = (f"Ear{index}_{side}" if index else f"Ear_{side}",)
    for target, candidates in aliases.items():
        if target not in targets:
            continue
        for candidate in candidates:
            if normalized(candidate) in sources:
                source = sources[normalized(candidate)]
                if source not in result:
                    result[source] = target
                break
    required = {"pelvis", "head", "arm_upper_L", "arm_upper_R", "arm_lower_L", "arm_lower_R",
                "hand_L", "hand_R", "leg_upper_L", "leg_upper_R", "leg_lower_L", "leg_lower_R",
                "ankle_L", "ankle_R"}
    missing = sorted(required - set(result.values()))
    if missing:
        raise ValueError("Automatic humanoid conversion cannot identify: " + ", ".join(missing))
    return result
