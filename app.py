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
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas

# --- BRANDING & PAGE SETUP ---
st.set_page_config(
    page_title="MassingPro | Architectural Context Generator", 
    page_icon="🏢",
    layout="wide"
)

# --- CSS Dark Studio ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1c24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CONCURRENCY & CACHING ---
@st.cache_resource
def get_lock():
    return threading.Lock()

generation_lock = get_lock()

@st.cache_resource(show_spinner=False)
def load_onnx_model():
    return ort.InferenceSession("depth_anything_v2_vits.onnx", providers=['CPUExecutionProvider'])

# --- 2. CORE UTILITIES ---
def unwarp_facade(image_pil, src_points, physical_width, physical_height, output_res=1024):
    image_cv = np.array(image_pil)
    aspect_ratio = physical_width / float(physical_height)
    
    if aspect_ratio > 1.0:
        out_w, out_h = output_res, int(output_res / aspect_ratio)
    else:
        out_h, out_w = output_res, int(output_res * aspect_ratio)
        
    src_pts = np.array([[p['x'], p['y']] for p in src_points], dtype="float32")
    dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_cv = cv2.warpPerspective(image_cv, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return Image.fromarray(warped_cv)

def process_depth_and_normals(image_pil, mask_data, session, normal_strength):
    image_cv = np.array(image_pil)
    orig_h, orig_w = image_cv.shape[:2]
    
    img_resized = cv2.resize(image_cv, (518, 518), interpolation=cv2.INTER_CUBIC)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    depth_out = session.run(None, {input_name: tensor})[0]
    
    depth_array = cv2.resize(np.squeeze(depth_out), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    d_min, d_max = depth_array.min(), depth_array.max()
    depth_normalized = (depth_array - d_min) / (d_max - d_min + 1e-8)
    
    if mask_data is not None:
        alpha_channel = mask_data[:, :, 3]
        binary_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
        if np.any(binary_mask):
            depth_normalized = cv2.inpaint(
                (depth_normalized * 255).astype(np.uint8), 
                binary_mask, 3, cv2.INPAINT_NS
            ).astype(np.float32) / 255.0

    depth_filtered = cv2.bilateralFilter(depth_normalized, d=9, sigmaColor=0.05, sigmaSpace=75)
    depth_16bit = (depth_filtered * 65535.0).astype(np.uint16)
    
    sobel_x = cv2.Sobel(depth_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_filtered, cv2.CV_64F, 0, 1, ksize=3)
    
    mag = np.sqrt((sobel_x * normal_strength)**2 + (sobel_y * normal_strength)**2 + 1.0)
    norm_x, norm_y, norm_z = -(sobel_x * normal_strength) / mag, -(sobel_y * normal_strength) / mag, 1.0 / mag
    
    normal_map = np.stack([
        ((norm_x + 1.0) * 127.5).astype(np.uint8),
        ((norm_y + 1.0) * 127.5).astype(np.uint8),
        ((norm_z + 1.0) * 127.5).astype(np.uint8)
    ], axis=2)
    
    return Image.fromarray(depth_16bit, mode='I;16'), Image.fromarray(normal_map, mode='RGB')

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [255]

def create_textured_plane(vertices, uvs, diffuse_img, normal_img, blank_color_rgba):
    faces = [[0, 1, 2], [0, 2, 3]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if diffuse_img is not None:
        pbr_mat = material.PBRMaterial(
            baseColorTexture=diffuse_img, normalTexture=normal_img,
            metallicFactor=0.0, roughnessFactor=0.8
        )
        mesh.visual = texture.TextureVisuals(uv=uvs, material=pbr_mat)
    else:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[blank_color_rgba]*2)
    return mesh

# --- 3. UI APPLICATION ---
st.title("MassingPro")
st.subheader("Architectural Facade-to-Box Pipeline")

faces = ["Front", "Back", "Left", "Right"]
if 'masks' not in st.session_state: st.session_state.masks = {f: None for f in faces}
if 'warped' not in st.session_state: st.session_state.warped = {f: None for f in faces}
if 'clicks' not in st.session_state: st.session_state.clicks = {f: [] for f in faces}

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/architecture.png", width=60)
    st.header("Project Parameters")
    dim_x = st.number_input("Width (X axis) m", 1.0, value=10.0, help="The building width along the primary facade.")
    dim_y = st.number_input("Depth (Y axis) m", 1.0, value=15.0, help="The depth of the building volume.")
    dim_z = st.number_input("Height (Z axis) m", 1.0, value=12.0, help="The total height from ground to parapet.")
    
    st.divider()
    st.header("Render Settings")
    blank_color = st.color_picker("Empty Face Material (ID)", "#2A2D35")
    n_strength = st.slider("Normal Mapping Intensity", 0.1, 5.0, 2.0)
    
    st.info("MassingPro uses 16-bit PNG displacement maps to prevent banding in V-Ray and Enscape.")

# Main Interface
st.markdown("---")
tabs = st.tabs([f" {f} Facade" for f in faces])
face_dims = {"Front": (dim_x, dim_z), "Back": (dim_x, dim_z), "Left": (dim_y, dim_z), "Right": (dim_y, dim_z)}
uv_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]]) 

for i, face in enumerate(faces):
    with tabs[i]:
        up_file = st.file_uploader(f"Upload {face} Elevation", type=["jpg", "png"], key=f"up_{face}")
        if up_file:
            raw = Image.open(up_file).convert("RGB")
            
            if st.session_state.warped[face] is None:
                st.markdown("#### 1. Rectify Perspective")
                st.caption("Identify the four corners of the facade: TL -> TR -> BR -> BL.")
                coords = streamlit_image_coordinates(raw, key=f"coord_{face}")
                
                if coords and coords not in st.session_state.clicks[face]:
                    st.session_state.clicks[face].append(coords)
                    
                st.write(f"Points Defined: {len(st.session_state.clicks[face])} / 4")
                
                if len(st.session_state.clicks[face]) == 4:
                    if st.button(f"Un-warp {face} to Scale", key=f"btn_{face}"):
                        w, h = face_dims[face]
                        st.session_state.warped[face] = unwarp_facade(raw, st.session_state.clicks[face], w, h)
                        st.rerun()
                if st.button("Reset Points", key=f"rst_{face}", help="Clear corners"):
                    st.session_state.clicks[face] = []
                    st.rerun()
            else:
                st.markdown("#### 2. Clean Geometry (Optional)")
                st.caption("Scribble over trees/cars to flatten them in the displacement map.")
                if st.button("Re-Crop Perspective", key=f"undo_{face}"):
                    st.session_state.warped[face], st.session_state.clicks[face] = None, []
                    st.rerun()
                
                warped_img = st.session_state.warped[face]
                canvas_w = min(1000, warped_img.width)
                canvas_h = int(warped_img.height * (canvas_w / warped_img.width))
                
                canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.4)", stroke_width=25, stroke_color="#FF0000",
                    background_image=warped_img, update_streamlit=True,
                    height=canvas_h, width=canvas_w,
                    drawing_mode="freedraw", key=f"canvas_{face}"
                )
                
                if canvas.image_data is not None:
                    st.session_state.masks[face] = cv2.resize(canvas.image_data, (warped_img.width, warped_img.height), interpolation=cv2.INTER_NEAREST)

st.markdown("---")

# --- 4. EXPORT ---
if st.session_state.warped["Front"] is None:
    st.info("ℹ️ Upload and rectify the Front Facade to begin.")
else:
    if st.button("BUILD MASSING PRO ASSET", type="primary", use_container_width=True):
        queue_placeholder = st.empty()
        if generation_lock.locked():
            queue_placeholder.warning("⏳ Queue Active: Another professional is currently processing an asset. Please hold...")
            
        generation_lock.acquire(blocking=True)
        queue_placeholder.empty() 
        
        try:
            with st.spinner("Executing Depth Estimation & UV Mapping..."):
                session = load_onnx_model()
                blank_rgba = hex_to_rgba(blank_color)
                meshes, displacements = [], {}
                
                plane_verts = {
                    "Front": np.array([[0, 0, dim_z], [dim_x, 0, dim_z], [dim_x, 0, 0], [0, 0, 0]]),
                    "Back":  np.array([[dim_x, dim_y, dim_z], [0, dim_y, dim_z], [0, dim_y, 0], [dim_x, dim_y, 0]]),
                    "Left":  np.array([[0, dim_y, dim_z], [0, 0, dim_z], [0, 0, 0], [0, dim_y, 0]]),
                    "Right": np.array([[dim_x, 0, dim_z], [dim_x, dim_y, dim_z], [dim_x, dim_y, 0], [dim_x, 0, 0]]),
                    "Top":   np.array([[0, dim_y, dim_z], [dim_x, dim_y, dim_z], [dim_x, 0, dim_z], [0, 0, dim_z]]),
                    "Bot":   np.array([[0, 0, 0], [dim_x, 0, 0], [dim_x, dim_y, 0], [0, dim_y, 0]])
                }
                
                for f_name in faces:
                    if st.session_state.warped[f_name]:
                        disp_img, norm_img = process_depth_and_normals(
                            st.session_state.warped[f_name], st.session_state.masks[f_name], session, n_strength
                        )
                        displacements[f_name] = disp_img
                        mesh = create_textured_plane(plane_verts[f_name], uv_coords, st.session_state.warped[f_name], norm_img, blank_rgba)
                    else:
                        mesh = create_textured_plane(plane_verts[f_name], uv_coords, None, None, blank_rgba)
                    meshes.append(mesh)
                    
                meshes.append(create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba))
                meshes.append(create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba))
                
                scene = trimesh.Scene(meshes)
                glb_data = scene.export(file_type='glb')
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("MassingPro_Asset.glb", glb_data)
                    for f_name, disp_img in displacements.items():
                        disp_bytes = io.BytesIO()
                        disp_img.save(disp_bytes, format='PNG')
                        zip_file.writestr(f"Displacement_{f_name}_16bit.png", disp_bytes.getvalue())
                
                st.success("✅ Compilation Successful.")
                st.download_button("📦 DOWNLOAD MASSINGPRO PACKAGE (.ZIP)", data=zip_buffer.getvalue(), file_name="MassingPro_Context.zip", mime="application/zip")
        finally:
            generation_lock.release()
