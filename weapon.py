"""Identify a native weapon branch that can follow a retargeted avatar grip."""


def weapon_binding(native_bones, mapping, names):
    parents={bone.name:bone.parent for bone in native_bones}
    reverse={role:name for name,role in mapping.items()}
    root=next((name for name in ('weaponPivot','weapon','weaponHier_JNT') if name in parents),None)
    if root is None:return None
    hand=next((role for role in ('hand_R','hand_L') if role in parents and role in reverse),None)
    if hand is None:return None
    def belongs(name):
        while name is not None:
            if name==root:return True
            name=parents.get(name)
        return False
    branch=[name for name in parents if belongs(name)]
    # Never move native body joints when correcting the weapon branch.
    if set(branch)&set(mapping.values()):
        raise ValueError('Native weapon branch contains mapped body bones; cannot align it independently')
    ancestor=parents[root]
    while ancestor is not None and ancestor!=hand:ancestor=parents.get(ancestor)
    hand_branch=[name for name in parents if name==hand or _descends_from(name,hand,parents)]
    return {'root':root,'native_hand':hand,'avatar_hand':names[reverse[hand]],'bones':branch,
            'root_follows_native_hand':ancestor==hand,'hand_bones':hand_branch}


def _descends_from(name,parent,parents):
    while name is not None:
        if name==parent:return True
        name=parents.get(name)
    return False
