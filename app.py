import streamlit as st
import base64
from io import BytesIO

# --- THE BULLETPROOF CANVAS PATCH ---
def universal_b64_encoder(image, *args, **kwargs):
    try:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return ""

import streamlit.elements.image as st_image
import streamlit_drawable_canvas
st_image.image_to_url = universal_b64_encoder
streamlit_drawable_canvas.image_to_url = universal_b64_encoder
# ------------------------------------

import numpy as np
import cv2
from PIL import Image, ImageDraw
import onnxruntime as ort
import trimesh
from trimesh.visual import texture, material
import io
import zipfile
import threading
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MassingPro | Architectural Context Generator", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_lock(): return threading.Lock()
generation_lock = get_lock()

@st.cache_resource(show_spinner=False)
def load_onnx_model():
    return ort.InferenceSession("depth_anything_v2_vits.onnx", providers=['CPUExecutionProvider'])

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

st.title("MassingPro")
faces = ["Front", "Back", "Left", "Right"]
for k in ['masks', 'warped', 'clicks']: 
    if k not in st.session_state: st.session_state[k] = {f: [] if k=='clicks' else None for f in faces}

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
                hud_img = raw.copy()
                draw = ImageDraw.Draw(hud_img)
                pts = st.session_state.clicks[face]
                for p in pts: draw.ellipse((p['x']-10, p['y']-10, p['x']+10, p['y']+10), fill="#ff4b4b", outline="white", width=2)
                if len(pts) > 1:
                    for j in range(len(pts)-1): draw.line([(pts[j]['x'], pts[j]['y']), (pts[j+1]['x'], pts[j+1]['y'])], fill="#ff4b4b", width=5)
                if len(pts) == 4: draw.line([(pts[3]['x'], pts[3]['y']), (pts[0]['x'], pts[0]['y'])], fill="#ff4b4b", width=5)
                
                st.write(f"Click 4 corners (TL, TR, BR, BL). Points: {len(pts)}/4")
                coords = streamlit_image_coordinates(hud_img, key=f"coord_{face}")
                if coords and (not pts or coords != pts[-1]) and len(pts) < 4:
                    st.session_state.clicks[face].append(coords)
                    st.rerun()
                
                col1, col2 = st.columns(2)
                if col1.button("Clear All", key=f"clr_{face}"): st.session_state.clicks[face] = []; st.rerun()
                if len(pts) > 0 and col2.button("Undo Last", key=f"und_{face}"): st.session_state.clicks[face].pop(); st.rerun()
                if len(pts) == 4 and st.button("Un-warp Elevation", key=f"uwp_{face}", type="primary"):
                    w, h = face_dims[face]
                    st.session_state.warped[face] = unwarp_facade(raw, pts, w, h)
                    st.rerun()
            else:
                st.success("✅ Perspective Rectified")
                w_img = st.session_state.warped[face]
                
                # NATIVE UI OVERRIDE: Force image display regardless of canvas background status
                st.image(w_img, caption=f"{face} Elevation (Unwarped)", use_column_width=True)
                
                if st.button("Edit Perspective", key=f"re_{face}"): 
                    st.session_state.warped[face] = None; st.rerun()
                
                st.markdown("##### 🖌️ Optional: Mask foreground objects (Trees/Cars) to flatten them.")
                canvas_w = min(1000, w_img.width)
                canvas_h = int(w_img.height * (canvas_w / w_img.width))
                canvas = st_canvas(fill_color="rgba(255, 0, 0, 0.3)", stroke_width=20, stroke_color="#FF0000",
                                  background_image=w_img, update_streamlit=True, height=canvas_h, width=canvas_w,
                                  drawing_mode="freedraw", key=f"canvas_v3_{face}")
                
                if canvas.image_data is not None:
                    st.session_state.masks[face] = cv2.resize(canvas.image_data, (w_img.width, w_img.height), interpolation=cv2.INTER_NEAREST)

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
                    normals[f] = norm
                    meshes.append(create_textured_plane(plane_verts[f], uv_coords, st.session_state.warped[f], norm, blank_rgba))
                else: meshes.append(create_textured_plane(plane_verts[f], uv_coords, None, None, blank_rgba))
            meshes.append(create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba))
            meshes.append(create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba))
            
            scene = trimesh.Scene(meshes)
            glb = scene.export(file_type='glb')
            buf = io.BytesIO()
            
            # --- PRO EXPORT PIPELINE: Giving you ALL the maps for Enscape/V-Ray ---
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("MassingPro_Geometry.glb", glb)
                for f, img in displacements.items():
                    # 1. Albedo (Diffuse)
                    albedo_buf = io.BytesIO(); st.session_state.warped[f].save(albedo_buf, format='JPEG')
                    zf.writestr(f"Maps/{f}_Albedo.jpg", albedo_buf.getvalue())
                    
                    # 2. 16-bit Displacement
                    disp_buf = io.BytesIO(); img.save(disp_buf, format='PNG')
                    zf.writestr(f"Maps/{f}_Displacement_16bit.png", disp_buf.getvalue())
                    
                    # 3. Normal Map
                    norm_buf = io.BytesIO(); normals[f].save(norm_buf, format='PNG')
                    zf.writestr(f"Maps/{f}_Normal.png", norm_buf.getvalue())

            st.success("✅ Compilation Successful. Extract maps and link in host software.")
            st.download_button("📦 DOWNLOAD PRO PACKAGE", data=buf.getvalue(), file_name="MassingPro_Asset.zip", mime="application/zip")
    finally: generation_lock.release()
