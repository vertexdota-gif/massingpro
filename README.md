# MassingPro

AI-powered architectural context massing generator. Upload facade photos, extract perspective, and export a fully textured 3D model ready for Rhino, SketchUp, or any render engine in one click.

---

## What it does

1. Upload photos of up to four building facades (Front, Back, Left, Right)
2. Click four corners to correct perspective distortion
3. Paint a mask over any artifacts (windows, vehicles, scaffolding)
4. Set building dimensions and hit **Build**

The app runs Depth-Anything-V2 to generate displacement and normal maps, then packages everything into a ZIP ready for your target software.

---

## Export formats

### Render Model — `.obj` + `.mtl` + `.matpkg`
Best for Enscape photorealistic rendering in Rhino.

- Flat ZIP structure — OBJ, MTL, and all textures sit in the root directory so Rhino resolves paths automatically
- MTL includes `map_Kd` (albedo) and `map_bump` (normal) — textures appear in the Rhino viewport with no extra steps
- Enscape `.matpkg` sidecars contain the full PBR schema (`BumpMapType: 3`, `ImageFade`, `UseColorChannel`) for physical displacement rendering
- `MassingPro_{id}_ApplyMaterials.py` — Rhino Python script that auto-sets all material texture slots on run

### Visual Model — `.glb`
Best for rapid viewport iteration in Rhino or SketchUp. Self-contained with embedded textures. Note: Rhino imports GLB as a Block Instance, which restricts Enscape material replacement.

### SketchUp / DAE — `.dae`
Best for SketchUp users who need textures visible in the viewport and accessible to render engines.

- Collada 1.4.1 format — SketchUp's native import path for textured geometry
- Albedo texture bound to `<diffuse>` — appears immediately in the SketchUp viewport on import
- Normal map included via `FCOLLADA` bump technique — picked up by V-Ray for SketchUp and Enscape for SketchUp
- Textures sit flat alongside the `.dae` file; relative paths resolve automatically on import
- Enscape `.matpkg` sidecars and displacement maps included in the ZIP for Enscape for SketchUp

---

## Rhino workflow (Render Model)

1. Extract the ZIP to a folder
2. Import the `.obj` into Rhino — textures appear in the viewport automatically
3. Open Enscape — albedo and normal maps render without any additional setup
4. **Optional:** Run `MassingPro_{id}_ApplyMaterials.py` via `Tools > Python Script > Run` to force-refresh material slots
5. **Optional:** Batch import the `.matpkg` files in the Enscape Material Editor for physical displacement

---

## SketchUp workflow (DAE)

1. Extract the ZIP to a folder
2. Go to **File > Import**, select the `.dae` file — textures load automatically from the same folder
3. The facade textures appear on the building surfaces in the SketchUp viewport immediately
4. For Enscape for SketchUp: open the Enscape Material Editor and import the `.matpkg` sidecars for displacement and PBR properties
5. For V-Ray for SketchUp: the normal map is already referenced in the DAE and will be read on scene open

---

## Auto perspective detection

After uploading a photo, click **🎯 Auto-Detect** to auto-fill the four corner points using edge and contour analysis — this is an optional shortcut. You can always ignore it and click the four corners manually on the canvas. If auto-detection lands incorrectly, drag the points to correct them or clear and re-click from scratch.

---

## Tech stack

| Layer | Library |
|---|---|
| Frontend | Streamlit + custom HTML/JS components |
| Depth AI | ONNX Runtime · Depth-Anything-V2 ViT-S |
| Geometry | trimesh · OpenCV |
| Export | zipfile · PIL |

---

## Local setup

```bash
pip install streamlit onnxruntime trimesh opencv-python pillow requests numpy
streamlit run app.py
```

Place `depth_anything_v2_vits.onnx` in the project root. Download from the [Depth-Anything-V2 releases](https://github.com/DepthAnything/Depth-Anything-V2).

---

## Material naming convention

All materials follow the pattern `MassingPro_{project_id}_{face}` (e.g. `MassingPro_482910_Front`). The project ID is randomised per build. The Enscape matpkg `"Name"` field matches this exactly to enable auto-linking by name.

---

## Changelog

### 2026-04-09
- **SketchUp / DAE export** — new export option generating a Collada 1.4.1 file with albedo textures bound to `<diffuse>` (viewport) and normal maps via FCOLLADA bump technique (V-Ray / Enscape for SketchUp). Displacement maps and `.matpkg` sidecars included in the ZIP.
- **Interactive 3D preview** — inline model-viewer panel renders a double-sided GLB preview directly in the app after each build, with expand/collapse toggle.

### 2026-04-05
- **Render Model export** — OBJ + MTL with `map_Kd` and `map_bump` for zero-step Rhino viewport textures and Enscape rendering.
- **Rhino auto-material script** — `MassingPro_{id}_ApplyMaterials.py` generated in every ZIP to force-apply texture slots via Rhino Python.
- **Auto perspective detection** — one-click corner detection using edge + contour analysis; points are draggable for refinement.

### 2026-04-04
- **Hybrid export pipeline** — GLB for viewport visibility paired with Enscape `.matpkg` sidecars for PBR rendering.

### 2026-04-03
- **Initial release** — core pipeline: perspective unwarp, Depth-Anything-V2 depth estimation, normal map generation, 16-bit displacement maps, and ZIP packaging.
- **Enscape `.matpkg` generation** — full PBR JSON schema with albedo, normal, and displacement texture slots.
- **Unique project IDs** — randomised per build to prevent material name collisions in Rhino/Enscape.
- **Custom HTML/JS canvas components** — zero-lag perspective point picker and mask painter.

---

## License

Apache License 2.0
