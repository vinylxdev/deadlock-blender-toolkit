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
6. Under **Build & Export**, choose your texture resolution and **Keep native gun** setting, then click **Build Retargeted VPK**.
7. Import the resulting VPK into your mod manager.

All avatar meshes must be bound to one armature and have accessible textures. Automatic material baking expects Principled BSDF materials. Builds run in a background Blender process; press **Esc** to cancel. Test the resulting mod in game.
