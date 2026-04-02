import streamlit as st
import base64
from io import BytesIO
import streamlit_drawable_canvas

# --- THE CORRECTED BASE64 INJECTION ---
def base64_image_encoder(image, *args, **kwargs):
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Hijack the exact reference inside the canvas module (No underscore)
streamlit_drawable_canvas.image_to_url = base64_image_encoder
# --------------------------------------

import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import trimesh
from trimesh.visual import texture, material
import io
import zipfile
import threading
from streamlit_drawable_canvas import st_canvas

# --- BRANDING & PAGE SETUP ---
st.set_page_config(page_title="MassingPro | Architectural Context Generator", page_icon="🏢", layout="wide")

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

def create_textured_plane(vertices, uvs, diffuse_img, normal_img, blank_rgba):
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if diffuse_img:
        mesh.visual = texture.TextureVisuals(uv=uvs, material=material.PBRMaterial(baseColorTexture=diffuse_img, normalTexture=normal_img, metallicFactor=0.0, roughnessFactor=0.8))
    else:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[blank_rgba]*2)
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
                st.info("📍 Click 4 corners (TL → TR → BR → BL). Use the trash can icon on the canvas to undo.")
                
                canvas_w = min(1000, raw.width)
                canvas_h = int(raw.height * (canvas_w / raw.width))
                
                # Using st_canvas for pure client-side point placement
                canvas_pts = st_canvas(
                    fill_color="rgba(255, 75, 75, 1)",
                    stroke_color="rgba(255, 75, 75, 1)",
                    background_image=raw,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="point",
                    point_display_radius=6,
                    key=f"canvas_pts_{face}"
                )
                
                if canvas_pts.json_data is not None and "objects" in canvas_pts.json_data:
                    objects = canvas_pts.json_data["objects"]
                    points = [{"x": obj["left"], "y": obj["top"]} for obj in objects]
                    
                    st.write(f"**Points Selected:** {len(points)} / 4")
                    
                    if len(points) == 4:
                        if st.button(f"Un-warp {face} Elevation", key=f"uwp_{face}", type="primary"):
                            # Scale coordinates back to original image size
                            sx = raw.width / canvas_w
                            sy = raw.height / canvas_h
                            scaled_pts = [{"x": p["x"] * sx, "y": p["y"] * sy} for p in points]
                            
                            w, h = face_dims[face]
                            st.session_state.warped[face] = unwarp_facade(raw, scaled_pts, w, h)
                            st.rerun()
            else:
                st.success("✅ Perspective Rectified")
                if st.button("Edit Perspective", key=f"re_{face}"): 
                    st.session_state.warped[face] = None; st.rerun()
                
                st.markdown("#### 2. Mask Occlusions (Optional)")
                w_img = st.session_state.warped[face]
                canvas_w = min(1000, w_img.width)
                canvas_h = int(w_img.height * (canvas_w / w_img.width))
                
                canvas_mask = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)", stroke_width=20, stroke_color="#FF0000",
                    background_image=w_img, update_streamlit=True, height=canvas_h, width=canvas_w,
                    drawing_mode="freedraw", key=f"canvas_mask_{face}"
                )
                
                if canvas_mask.image_data is not None:
                    st.session_state.masks[face] = cv2.resize(canvas_mask.image_data, (w_img.width, w_img.height), interpolation=cv2.INTER_NEAREST)

st.divider()

if st.session_state.warped["Front"] and st.button("BUILD MASSING PRO ASSET", type="primary", use_container_width=True):
    queue_placeholder = st.empty()
    if generation_lock.locked(): queue_placeholder.warning("⏳ Queue Active: Processing another request...")
    generation_lock.acquire(blocking=True)
    queue_placeholder.empty()
    try:
        with st.spinner("Processing AI Maps & Packaging GLB..."):
            session = load_onnx_model()
            blank_rgba = [int(blank_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [255]
            meshes, displacements = [], {}
            plane_verts = {
                "Front": np.array([[0, 0, dim_z], [dim_x, 0, dim_z], [dim_x, 0, 0], [0, 0, 0]]),
                "Back":  np.array([[dim_x, dim_y, dim_z], [0, dim_y, dim_z], [0, dim_y, 0], [dim_x, dim_y, 0]]),
                "Left":  np.array([[0, dim_y, dim_z], [0, 0, dim_z], [0, 0, 0], [0, dim_y, 0]]),
                "Right": np.array([[dim_x, 0, dim_z], [dim_x, dim_y, dim_z], [dim_x, dim_y, 0], [dim_x, 0, 0]]),
                "Top":   np.array([[0, dim_y, dim_z], [dim_x, dim_y, dim_z], [dim_x, 0, dim_z], [0, 0, dim_z]]),
                "Bot":   np.array([[0, 0, 0], [dim_x, 0, 0], [dim_x, dim_y, 0], [0, dim_y, 0]])
            }
            for f in faces:
                if st.session_state.warped[f]:
                    disp, norm = process_depth_and_normals(st.session_state.warped[f], st.session_state.masks[f], session, n_strength)
                    displacements[f] = disp
                    meshes.append(create_textured_plane(plane_verts[f], uv_coords, st.session_state.warped[f], norm, blank_rgba))
                else: meshes.append(create_textured_plane(plane_verts[f], uv_coords, None, None, blank_rgba))
            meshes.append(create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba))
            meshes.append(create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba))
            
            scene = trimesh.Scene(meshes)
            glb = scene.export(file_type='glb')
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("MassingPro_Asset.glb", glb)
                for f, img in displacements.items():
                    img_buf = io.BytesIO(); img.save(img_buf, format='PNG')
                    zf.writestr(f"Displacement_{f}_16bit.png", img_buf.getvalue())
            st.success("✅ Compilation Successful.")
            st.download_button("📦 DOWNLOAD PACKAGE", data=buf.getvalue(), file_name="MassingPro_Context.zip", mime="application/zip")
    finally: generation_lock.release()
