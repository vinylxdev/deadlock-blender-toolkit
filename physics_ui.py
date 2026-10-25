"""Compact chain settings: everyday motion, one curve at a time, and advanced controls."""
from .physics_curve import editor

CURVE_ITEMS = (
    ('stiffness', 'Return to pose', 'Attraction to the animated pose'),
    ('damping', 'Damping', 'How quickly motion settles'),
    ('gravity', 'Gravity', 'Gravity along the chain'),
    ('twist_resistance', 'Twist resistance', 'Cloth torsion spring strength'),
    ('motion_limit', 'Motion radius', 'Distance allowed from the animated pose'),
    ('motion_limit_strength', 'Limit strength', 'How firmly the motion radius is enforced'),
)


def slider(layout, operator, key):
    row = layout.split(factor=0.82, align=True)
    row.column(align=True).prop(operator, key, slider=True)
    row.column(align=True).prop(operator, 'use_' + key + '_curve', text='Curve', icon='FCURVE', toggle=True)


def draw_curve(layout, operator):
    layout.prop(operator, 'curve_setting', text='Setting')
    key = operator.curve_setting
    box = layout.box()
    if key.startswith('motion_limit'):
        if not operator.use_motion_limits:
            layout.label(text='Enable Motion limits in Motion to use this profile.', icon='INFO')
        elif operator.lock_twist:
            layout.label(text='Turn off Lock twist in Motion to use limits.', icon='INFO')
        box.enabled = operator.use_motion_limits and not operator.lock_twist
    elif key == 'twist_resistance' and operator.lock_twist:
        layout.label(text='Twist resistance affects the preview only.', icon='INFO')
    box.prop(operator, key, text='Base value', slider=True)
    box.prop(operator, 'use_' + key + '_curve', text='Use curve')
    graph = box.column()
    graph.enabled = getattr(operator, 'use_' + key + '_curve')
    tree = editor(getattr(operator, key + '_curve_name'))
    if tree:
        graph.template_curve_mapping(tree.nodes[0], 'mapping', type='NONE')
    else:
        graph.label(text='Reopen chain settings to edit the curve.')
    legend = graph.row()
    legend.label(text='Root')
    right = legend.row()
    right.alignment = 'RIGHT'
    right.label(text='Tip')
    box.label(text='Curve height multiplies the base value (0 to 1).')


def draw_dialog(layout, operator, rig):
    chain = layout.box()
    if rig:
        chain.prop_search(operator, 'root', rig.data, 'bones', text='Root')
        root = rig.data.bones.get(operator.root)
        if root:
            count = 1 + len(root.children_recursive)
            summary = f'{count} bone' + ('s' if count != 1 else '')
            summary += f' · {len(root.children)} branches' if len(root.children) > 1 else ' · Root stays attached'
            chain.label(text=summary, icon='BONE_DATA')
    else:
        chain.prop(operator, 'root')
    tabs = layout.row()
    tabs.scale_y = 1.15
    tabs.prop(operator, 'settings_tab', expand=True)

    if operator.settings_tab == 'MOTION':
        movement = layout.box()
        movement.label(text='Movement', icon='PHYSICS')
        for key in ('stiffness', 'damping', 'gravity'):
            slider(movement, operator, key)
        movement.separator()
        movement.prop(operator, 'lock_twist', text='Lock twist (swing only)')
        if operator.lock_twist:
            movement.label(text='Allows bending without simulated roll.', icon='INFO')

        limits = layout.box()
        header = limits.row()
        header.enabled = not operator.lock_twist or operator.use_motion_limits
        header.prop(operator, 'use_motion_limits', text='Motion limits')
        if operator.lock_twist:
            limits.label(text='Turn off Lock twist to use motion limits.', icon='INFO')
        elif operator.use_motion_limits:
            slider(limits, operator, 'motion_limit')
            slider(limits, operator, 'motion_limit_strength')
        else:
            limits.label(text='Keep movement close to the animated pose.')
        collision = layout.box()
        header = collision.row()
        header.enabled = not operator.lock_twist or operator.auto_collision
        header.prop(operator, 'auto_collision')
        if operator.lock_twist:
            collision.label(text='Turn off Lock twist to use body collisions.', icon='INFO')
        elif operator.auto_collision:
            collision.label(text='Body capsules fit automatically when previewing or building.')
        layout.label(text='Edit enabled profiles in the Curves tab.', icon='FCURVE')

    elif operator.settings_tab == 'CURVES':
        draw_curve(layout, operator)

    else:
        attachment = layout.box()
        attachment.label(text='Attachment', icon='CONSTRAINT_BONE')
        attachment.prop(operator, 'lock_root_rotation')
        attachment.prop(operator, 'anchor_child_roots')
        if operator.anchor_child_roots:
            attachment.label(text='Strand bases stay attached; lower bones simulate.')
        rotation = layout.box()
        rotation.label(text='Rotation stability', icon='ORIENTATION_LOCAL')
        slider(rotation, operator, 'twist_resistance')
        rotation.prop(operator, 'stable_rotation_frames')
        if operator.lock_twist:
            rotation.label(text='Twist resistance affects the preview only.', icon='INFO')

