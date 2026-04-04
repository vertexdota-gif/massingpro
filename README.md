# MassingPro

AI-powered architectural context massing generator. Upload facade photos, extract perspective, and export a fully textured 3D model ready for Rhino + Enscape in one click.

---

## What it does

1. Upload photos of up to four building facades (Front, Back, Left, Right)
2. Click four corners to correct perspective distortion
3. Paint a mask over any artifacts (windows, vehicles, scaffolding)
4. Set building dimensions and hit **Build**

The app runs Depth-Anything-V2 to generate displacement and normal maps, then packages everything into a ZIP ready to drop into Rhino.

---

## Export formats

### Render Model — `.obj` + `.mtl` + `.matpkg`
Best for Enscape photorealistic rendering.

- Flat ZIP structure — OBJ, MTL, and all textures sit in the root directory so Rhino resolves paths automatically
- MTL includes `map_Kd` (albedo) and `map_bump` (normal) — textures appear in the Rhino viewport with no extra steps
- Enscape `.matpkg` sidecars contain the full PBR schema (`BumpMapType: 3`, `ImageFade`, `UseColorChannel`) for physical displacement rendering
- `MassingPro_{id}_ApplyMaterials.py` — Rhino Python script that auto-sets all material texture slots on run

### Visual Model — `.glb`
Best for rapid viewport iteration in Rhino or SketchUp. Self-contained with embedded textures. Note: Rhino imports GLB as a Block Instance, which restricts Enscape material replacement.

---

## Rhino workflow (Render Model)

1. Extract the ZIP to a folder
2. Import the `.obj` into Rhino — textures appear in the viewport automatically
3. Open Enscape — albedo and normal maps render without any additional setup
4. **Optional:** Run `MassingPro_{id}_ApplyMaterials.py` via `Tools > Python Script > Run` to force-refresh material slots
5. **Optional:** Batch import the `.matpkg` files in the Enscape Material Editor for physical displacement

---

## Street View integration

Add your Google Maps API key to `.streamlit/secrets.toml`:

```toml
GOOGLE_MAPS_API_KEY = "your_key_here"
```

Or paste it directly into the sidebar at runtime. Requires **Street View Static API** enabled in your Google Cloud project.

---

## Auto perspective detection

Click **🎯 Auto-Detect** after uploading a photo to auto-fill the four corner points using edge and contour analysis. Drag any point to refine before extracting.

---

## Tech stack

| Layer | Library |
|---|---|
| Frontend | Streamlit + custom HTML/JS components |
| Depth AI | ONNX Runtime · Depth-Anything-V2 ViT-S |
| Geometry | trimesh · OpenCV |
| Export | zipfile · PIL · rhino3dm (optional) |

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

## License

Apache License 2.0
