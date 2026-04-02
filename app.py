import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import trimesh
from trimesh.visual import texture, material
import io
import zipfile
import threading
import random
import base64
import os
import streamlit.components.v1 as components

# --- 0. AUTO-BUILD CUSTOM UI COMPONENT ---
# This bypasses all of Streamlit's broken libraries by creating our own bulletproof UI
UI_DIR = "custom_ui"
os.makedirs(UI_DIR, exist_ok=True)
html_content = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/streamlit-component-lib@1.3.0/dist/streamlit.js"></script>
  <style>
    body { margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; font-family: sans-serif; }
    #container { position: relative; display: inline-block; }
    canvas { cursor: crosshair; max-width: 1000px; height: auto; }
    #drawLayer { position: absolute; top: 0; left: 0; pointer-events: none; }
    .controls { margin-top: 10px; display: flex; gap: 10px; }
    button { background: #ff4b4b; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }
    p { color: white; margin: 0; padding: 8px 0; }
  </style>
</head>
<body>
  <div id="container">
    <canvas id="bgLayer"></canvas>
    <canvas id="drawLayer"></canvas>
  </div>
  <div class="controls" id="controls"></div>

  <script>
    const bgCanvas = document.getElementById("bgLayer");
    const bgCtx = bgCanvas.getContext("2d");
    const drawCanvas = document.getElementById("drawLayer");
    const drawCtx = drawCanvas.getContext("2d");
    const controls = document.getElementById("controls");

    let mode = "points";
    let points = [];
    let isDrawing = false;
    let img = new Image();

    function onRender(event) {
      if (img.src) return; // Only load once
      const args = event.detail.args;
      mode = args.mode;
      img.src = args.image_b64;

      img.onload = () => {
        bgCanvas.width = img.width; bgCanvas.height = img.height;
        drawCanvas.width = img.width; drawCanvas.height = img.height;
        bgCtx.drawImage(img, 0, 0);
        
        // Match CSS sizing for alignment
        const displayWidth = Math.min(1000, img.width);
        bgCanvas.style.width = displayWidth + "px";
        drawCanvas.style.width = displayWidth + "px";
        
        updateControls();
        Streamlit.setFrameHeight(bgCanvas.getBoundingClientRect().height + 60);
      }
    }

    function redrawPoints() {
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      drawCtx.fillStyle = "#ff4b4b"; drawCtx.strokeStyle = "#ff4b4b"; drawCtx.lineWidth = 4;
      points.forEach((p, i) => {
        drawCtx.beginPath(); drawCtx.arc(p.x, p.y, 10, 0, Math.PI * 2); drawCtx.fill();
        if (i > 0) { drawCtx.beginPath(); drawCtx.moveTo(points[i-1].x, points[i-1].y); drawCtx.lineTo(p.x, p.y); drawCtx.stroke(); }
      });
      if (points.length === 4) { drawCtx.beginPath(); drawCtx.moveTo(points[3].x, points[3].y); drawCtx.lineTo(points[0].x, points[0].y); drawCtx.stroke(); }
    }

    function updateControls() {
      if (mode === "points") {
        controls.innerHTML = `<p>📍 Points: ${points.length}/4</p>` +
          (points.length > 0 ? `<button onclick="undoPoint()">Undo</button>` : "") +
          (points.length === 4 ? `<button onclick="sendData()">Un-Warp Perspective</button>` : "");
      } else {
        controls.innerHTML = `<button onclick="sendData()">Confirm Mask</button> <button onclick="clearMask()">Clear</button>`;
      }
    }

    bgCanvas.addEventListener("mousedown", (e) => {
      const rect = bgCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (bgCanvas.width / rect.width);
      const y = (e.clientY - rect.top) * (bgCanvas.height / rect.height);

      if (mode === "points" && points.length < 4) {
        points.push({x, y});
        redrawPoints();
        updateControls();
      } else if (mode === "mask") {
        isDrawing = true;
        drawCtx.fillStyle = "rgba(255, 0, 0, 0.5)";
        drawCtx.beginPath(); drawCtx.arc(x, y, 20, 0, Math.PI * 2); drawCtx.fill();
      }
    });

    bgCanvas.addEventListener("mousemove", (e) => {
      if (!isDrawing || mode !== "mask") return;
      const rect = bgCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (bgCanvas.width / rect.width);
      const y = (e.clientY - rect.top) * (bgCanvas.height / rect.height);
      drawCtx.beginPath(); drawCtx.arc(x, y, 20, 0, Math.PI * 2); drawCtx.fill();
    });

    window.addEventListener("mouseup", () => { isDrawing = false; });

    window.undoPoint = () => { points.pop(); redrawPoints(); updateControls(); }
    window.clearMask = () => { drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height); }
    window.sendData = () => {
      if (mode === "points") Streamlit.setComponentValue({type: "points", data: points});
      if (mode === "mask") Streamlit.setComponentValue({type: "mask", data: drawCanvas.toDataURL()});
    }

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
  </script>
</body>
</html>
"""
with open(os.path.join(UI_DIR, "index.html"), "w", encoding="utf-8") as f: f.write(html_content)
massingpro_ui = components.declare_component("massingpro_ui", path=UI_DIR)

def pil_to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
# --------------------------------------------------------

# --- BRANDING & PAGE SETUP ---
st.set_page_config(page_title="MassingPro | Context Generator", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CONCURRENCY & CACHING ---
@st.cache_resource
def get_lock(): return threading.Lock()
generation_lock = get_lock()

@st.cache_resource(show_spinner=False)
def load_onnx_model():
    return ort.InferenceSession("depth_anything_v2_vits.onnx", providers=['CPUExecutionProvider'])

# --- 2. CORE UTILITIES ---
def unwarp_facade(image_pil, src_points, physical_width, physical_height, output_res=1024):
    image_cv = np.array(image_pil)
    aspect_ratio = physical_width / float(physical_height)
    out_w, out_h = (output_res, int(output_res / aspect_ratio)) if aspect_ratio > 1.0 else (int(output_res * aspect_ratio), output_res)
    src_pts = np.array([[p['x'], p['y']] for p in src_points], dtype="float32")
    dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_cv = cv2.warpPerspective(image_cv, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return Image.fromarray(warped_cv)

def process_depth_and_normals(image_pil, mask_data, session, normal_strength):
    image_cv = np.array(image_pil)
    orig_h, orig_w = image_cv.shape[:2]
    img_resized = cv2.resize(image_cv, (518, 518), interpolation=cv2.INTER_CUBIC)
    img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0).astype(np.float32)
    depth_out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    depth_array = cv2.resize(np.squeeze(depth_out), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
    
    if mask_data is not None:
        binary_mask = np.where(mask_data[:, :, 3] > 0, 255, 0).astype(np.uint8)
        if np.any(binary_mask):
            depth_normalized = cv2.inpaint((depth_normalized * 255).astype(np.uint8), binary_mask, 3, cv2.INPAINT_NS).astype(np.float32) / 255.0

    depth_filtered = cv2.bilateralFilter(depth_normalized, d=9, sigmaColor=0.05, sigmaSpace=75)
    sobel_x, sobel_y = cv2.Sobel(depth_filtered, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(depth_filtered, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt((sobel_x * normal_strength)**2 + (sobel_y * normal_strength)**2 + 1.0)
    normal_map = np.stack([((-(sobel_x * normal_strength) / mag + 1.0) * 127.5), (((-(sobel_y * normal_strength) / mag) + 1.0) * 127.5), ((1.0 / mag) * 255)], axis=2).astype(np.uint8)
    return Image.fromarray((depth_filtered * 65535.0).astype(np.uint16), mode='I;16'), Image.fromarray(normal_map, mode='RGB')

def create_textured_plane(vertices, uvs, diffuse_img, normal_img, blank_rgba, face_name, project_id):
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if face_name == "Bot":
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[blank_rgba]*2)
        return mesh
    if diffuse_img:
        mat = material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}", baseColorTexture=diffuse_img, normalTexture=normal_img, metallicFactor=0.0, roughnessFactor=0.8)
        mesh.visual = texture.TextureVisuals(uv=uvs, material=mat)
    else:
        mat = material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}_Blank", baseColorFactor=blank_rgba)
        mesh.visual = texture.TextureVisuals(uv=uvs, material=mat)
    return mesh

# --- 3. UI APPLICATION ---
st.title("MassingPro")
faces = ["Front", "Back", "Left", "Right"]
for k in ['masks', 'warped']: 
    if k not in st.session_state: st.session_state[k] = {f: None for f in faces}

with st.sidebar:
    st.header("Project Dimensions")
    dim_x = st.number_input("Width (X) m", 1.0, value=10.0)
    dim_y = st.number_input("Depth (Y) m", 1.0, value=15.0)
    dim_z = st.number_input("Height (Z) m", 1.0, value=12.0)
    blank_color = st.color_picker("Empty Face Color", "#2A2D35")
    n_strength = st.slider("Normal Intensity", 0.1, 5.0, 2.0)

tabs = st.tabs([f" {f} Facade" for f in faces])
face_dims = {"Front": (dim_x, dim_z), "Back": (dim_x, dim_z), "Left": (dim_y, dim_z), "Right": (dim_y, dim_z)}
uv_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]]) 

for i, face in enumerate(faces):
    with tabs[i]:
        up_file = st.file_uploader(f"Upload {face}", type=["jpg", "png"], key=f"up_{face}")
        if up_file:
            raw = Image.open(up_file).convert("RGB")
            
            if st.session_state.warped[face] is None:
                st.markdown("#### 1. Rectify Perspective")
                st.caption("Click 4 corners (TL → TR → BR → BL)")
                
                # Render custom UI component in "points" mode
                result = massingpro_ui(mode="points", image_b64=pil_to_b64(raw), key=f"ui_pts_{face}")
                
                if result and result.get("type") == "points":
                    pts = result["data"]
                    w, h = face_dims[face]
                    st.session_state.warped[face] = unwarp_facade(raw, pts, w, h)
                    st.rerun()
            else:
                st.success("✅ Perspective Rectified")
                if st.button("Edit Perspective", key=f"re_{face}"): 
                    st.session_state.warped[face] = None; st.rerun()
                
                st.markdown("#### 2. Mask Occlusions (Optional)")
                st.caption("Draw over trees/cars. Click 'Confirm Mask' to apply.")
                w_img = st.session_state.warped[face]
                
                # Render custom UI component in "mask" mode
                result = massingpro_ui(mode="mask", image_b64=pil_to_b64(w_img), key=f"ui_mask_{face}")
                
                if result and result.get("type") == "mask":
                    # Convert base64 mask back to OpenCV numpy array
                    encoded_data = result["data"].split(',')[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    st.session_state.masks[face] = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    st.success("Mask Saved!")

st.divider()

if st.session_state.warped["Front"] and st.button("BUILD MASSING PRO ASSET", type="primary", use_container_width=True):
    queue_placeholder = st.empty()
    if generation_lock.locked(): queue_placeholder.warning("⏳ Queue Active: Processing another request...")
    generation_lock.acquire(blocking=True)
    queue_placeholder.empty()
    try:
        with st.spinner("Processing AI Maps & Packaging Full Pipeline Asset..."):
            session = load_onnx_model()
            blank_rgba = [int(blank_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [255]
            meshes, displacements, normals = [], {}, {}
            project_id = str(random.randint(1000, 999999))
            
            plane_verts = {
                "Front": np.array([[0, dim_z, 0], [dim_x, dim_z, 0], [dim_x, 0, 0], [0, 0, 0]]),
                "Back":  np.array([[dim_x, dim_z, -dim_y], [0, dim_z, -dim_y], [0, 0, -dim_y], [dim_x, 0, -dim_y]]),
                "Left":  np.array([[0, dim_z, 0], [0, dim_z, -dim_y], [0, 0, -dim_y], [0, 0, 0]]),
                "Right": np.array([[dim_x, dim_z, -dim_y], [dim_x, dim_z, 0], [dim_x, 0, 0], [dim_x, 0, -dim_y]]),
                "Top":   np.array([[0, dim_z, 0], [dim_x, dim_z, 0], [dim_x, dim_z, -dim_y], [0, dim_z, -dim_y]]),
                "Bot":   np.array([[0, 0, -dim_y], [dim_x, 0, -dim_y], [dim_x, 0, 0], [0, 0, 0]])
            }
            
            for f in faces:
                if st.session_state.warped[f]:
                    disp, norm = process_depth_and_normals(st.session_state.warped[f], st.session_state.masks[f], session, n_strength)
                    displacements[f] = disp
                    normals[f] = norm
                    meshes.append(create_textured_plane(plane_verts[f], uv_coords, st.session_state.warped[f], norm, blank_rgba, f, project_id))
                else: meshes.append(create_textured_plane(plane_verts[f], uv_coords, None, None, blank_rgba, f, project_id))
            meshes.append(create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba, "Top", project_id))
            meshes.append(create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba, "Bot", project_id))
            
            scene = trimesh.Scene(meshes)
            glb = scene.export(file_type='glb')
            buf = io.BytesIO()
            
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"MassingPro_{project_id}.glb", glb)
                for f, img in displacements.items():
                    albedo_buf = io.BytesIO(); st.session_state.warped[f].save(albedo_buf, format='JPEG')
                    zf.writestr(f"Maps/{project_id}_{f}_Albedo.jpg", albedo_buf.getvalue())
                    disp_buf = io.BytesIO(); img.save(disp_buf, format='PNG')
                    zf.writestr(f"Maps/{project_id}_{f}_Displacement_16bit.png", disp_buf.getvalue())
                    norm_buf = io.BytesIO(); normals[f].save(norm_buf, format='PNG')
                    zf.writestr(f"Maps/{project_id}_{f}_Normal.png", norm_buf.getvalue())

            st.success(f"✅ Compilation Successful. Project ID: {project_id}")
            st.download_button("📦 DOWNLOAD PRO PACKAGE", data=buf.getvalue(), file_name=f"MassingPro_{project_id}.zip", mime="application/zip")
    finally: generation_lock.release()
