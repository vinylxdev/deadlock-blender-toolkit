# Deadlock Blender Toolkit

A Blender add-on for turning rigged avatars into Deadlock hero replacement mods. Preview your avatar on a hero's animations, configure accessory physics, bake materials, and build a VPK while keeping the source avatar intact.

Avatar rigs need to be supported manually, currently supported avatars that I have tested:
- Hyenid
- Novabeast
- Rexouium
I will add more supported types when I get the time.

## Requirements

- **Blender 5.2**.
- **Deadlock**, installed locally with access to `game/citadel`.
- **[SourceIO](https://github.com/REDxEYE/SourceIO)**, installed and enabled. Use a version supporting current Deadlock resources, including KV3 types 26/27.
- **[Source 2 Viewer CLI](https://github.com/ValveResourceFormat/ValveResourceFormat/releases)** for your operating system. Select the CLI executable, rather than the graphical viewer.
- **[Reduced CSDK 12](https://deadlockmodding.pages.dev/modding-tools/csdk-12)** for compiling models and materials.

On Windows, select `Source2Viewer-CLI.exe` and the SDK containing `resourcecompiler.exe`. Wine is only required on Linux, where both `wine` and `winepath` must be on `PATH`. Linux with Wine is the validated build environment; native Windows execution still needs an end-to-end build test. No separate Python installation is required to use the add-on.

If an older Viewer cannot decode a hero's KV3 data, extraction retries with a compatible encoding in a separate copy. If its animation graph also cannot be read, extraction uses embedded animations. Original retail resources and runtime animation bindings are preserved. Failed extraction cannot reuse a stale ModelDoc as successful output.

## Install

1. Download this repository using **Code → Download ZIP** or use the zip from Releases.
2. In Blender, open **Edit → Preferences → Add-ons → Install from Disk** and select the ZIP.
3. Enable **Deadlock Blender Toolkit** and **SourceIO**.
4. Open the 3D View sidebar with **N**, then select **Deadlock Toolkit**.

## Use

1. Open a rigged avatar and choose its **Model collection** under **Setup**.
2. Under **Setup → Directories**, set **Game folder** to Deadlock's `game/citadel`, choose an **Export folder**, and set **Reduced CSDK** and **Viewer CLI**.
3. Click **Refresh Heroes** and choose the hero to replace.
4. Under **Preview & Retarget**, choose **Preserve avatar**, then click **Retarget Original Rig Preview**.
5. Use **Preview Idle Pose** to inspect the result. Configure accessory chains under **Physics** and test them with **Preview Physics**.
6. Under **Build & Export**, choose your texture resolution and **Keep native gun** setting, click **Build Retargeted VPK**, and choose the VPK's location and filename in the save dialog.
7. Import the resulting VPK into your mod manager.

All avatar meshes must be bound to one armature and have accessible textures. Automatic material baking expects Principled BSDF materials. Builds run in a background Blender process; press **Esc** to cancel. Test the resulting mod in game.

Retargeted builds bake the first frame of the selected hero's idle into its respawn and rebirth countdown animations. This keeps the avatar posed while waiting to respawn, including when the gameplay animation graph is inactive.

The physics editor has three tabs: **Motion** for movement and motion limits, **Curves** for editing one profile at a time, and **Advanced** for attachment and rotation stability. Each slider's **Curve** button enables its profile; select the setting in **Curves** to edit the shape. **Apply** saves the chain settings.

Enable **Automatic collision fitting** in a chain's **Motion** tab to fit body capsules from the avatar's weighted rest mesh. Capsules follow the avatar's mapped torso, head, arm and leg bones. Fitting excludes accessory chains, runs once when starting Preview Physics or building, and affects only chains with the option enabled. The option saves with physics presets and requires **Lock twist** off. The capsules approximate the body surface; they do not provide triangle-level, world or accessory-to-accessory collision. Tight motion limits can prevent a chain from clearing a collider. Existing chains keep it disabled.

Each physics slider has an optional curve: **Return to pose**, **Damping**, **Gravity**, and **Twist resistance**. The graph's left edge is the selected root, its right edge is each branch's last bone, and its height multiplies that slider's value. For a firm base and loose tip, enable the Return to pose curve, keep the left side at 1, and lower the right side, for example to 0.25. Positions follow distances between bone heads in the rest pose; shared sections use their longest remaining branch. Each curve is independent. Disabled curves and flat curves at 1 preserve uniform settings.

Curves save with physics presets and affect Preview Physics and game export. With **Lock twist** enabled, Return to pose, Damping, and Gravity curves also control the native jiggle bones; Twist resistance applies to the cloth preview only. Preview remains an approximation of game physics.

Enable **Limit motion** on a cloth chain to bound its movement around the animated pose. **Motion radius** is a distance (stored in meters); its optional curve varies the radius from base to tip. **Limit strength** controls enforcement: 1 is a hard bound, lower values allow overshoot, and 0 disables the limit. It also has an optional curve. A radius of 0 disables that bone's limit rather than pinning it. Limits save with presets, run in Preview Physics, and export as Source 2 stray-radius constraints. Turn off **Lock twist** to use them. Existing chains keep limits disabled until you enable them.
