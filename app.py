import streamlit as st
import streamlit.components.v1 as components
import os
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
import json

# --- DYNAMIC CUSTOM COMPONENTS (ZERO-LAG UI) ---
PTS_DIR = "pts_frontend"
os.makedirs(PTS_DIR, exist_ok=True)
with open(f"{PTS_DIR}/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <body style="margin:0; padding:0; background: transparent; color: white; font-family: sans-serif;">
      <div id="root">
          <p style="margin-bottom:8px;">📍 Click 4 corners (TL &rarr; TR &rarr; BR &rarr; BL).</p>
          <canvas id="mycanvas" style="cursor:crosshair;border:1px solid #ff4b4b;border-radius:4px;"></canvas><br>
          <button id="btnClear" style="margin-top:8px;padding:8px 16px;background:#333;color:white;border:none;border-radius:4px;cursor:pointer;margin-right:8px;">Clear</button>
          <button id="btnSend" style="margin-top:8px;padding:8px 16px;background:#ff4b4b;color:white;border:none;border-radius:4px;cursor:pointer;">Extract Perspective</button>
      </div>
      <script>
        function send(type, data) { window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data), "*"); }
        send("streamlit:componentReady", {apiVersion: 1});
        let initialized = false; let points = [];
        window.addEventListener("message", function(event) {
            if (event.data.type === "streamlit:render" && !initialized) {
                initialized = true; const args = event.data.args;
                const canvas = document.getElementById('mycanvas');
                canvas.width = args.canvas_w; canvas.height = args.canvas_h;
                send("streamlit:setFrameHeight", {height: args.canvas_h + 100});
                const ctx = canvas.getContext('2d'); const img = new Image();
                img.src = 'data:image/png;base64,' + args.img_b64;
                img.onload = () => ctx.drawImage(img, 0, 0);
                function drawState() {
                    ctx.drawImage(img, 0, 0); ctx.fillStyle = '#ff4b4b'; ctx.strokeStyle = '#ff4b4b'; ctx.lineWidth = 3;
                    for (let i=0; i<points.length; i++) { ctx.beginPath(); ctx.arc(points[i].x, points[i].y, 6, 0, Math.PI*2); ctx.fill(); ctx.stroke(); }
                    if (points.length > 1) {
                        ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
                        for (let i=1; i<points.length; i++) ctx.lineTo(points[i].x, points[i].y);
                        if (points.length === 4) ctx.lineTo(points[0].x, points[0].y);
                        ctx.stroke();
                    }
                }
                canvas.addEventListener('mousedown', e => { if (points.length >= 4) return; const r = canvas.getBoundingClientRect(); points.push({x: e.clientX - r.left, y: e.clientY - r.top}); drawState(); });
                document.getElementById('btnClear').onclick = () => { points = []; drawState(); };
                document.getElementById('btnSend').onclick = () => {
                    if (points.length !== 4) return alert('Please select exactly 4 points.');
                    const scaleX = args.raw_w / args.canvas_w; const scaleY = args.raw_h / args.canvas_h;
                    const finalPts = points.map(p => ({x: p.x * scaleX, y: p.y * scaleY}));
                    send("streamlit:setComponentValue", {value: finalPts});
                };
            }
        });
      </script>
    </body>
    </html>
    """)
st_pts_picker = components.declare_component("pts_picker", path=PTS_DIR)

MASK_DIR = "mask_frontend"
os.makedirs(MASK_DIR, exist_ok=True)
with open(f"{MASK_DIR}/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <body style="margin:0; padding:0; background: transparent; color: white; font-family: sans-serif;">
      <div id="root">
          <canvas id="mycanvas" style="cursor:crosshair;border:1px solid #ff4b4b;border-radius:4px;"></canvas><br>
          <button id="btnClear" style="margin-top:8px;padding:8px 16px;background:#333;color:white;border:none;border-radius:4px;cursor:pointer;margin-right:8px;">Clear Brush</button>
          <button id="btnSend" style="margin-top:8px;padding:8px 16px;background:#ff4b4b;color:white;border:none;border-radius:4px;cursor:pointer;">Save Mask</button>
      </div>
      <script>
        function send(type, data) { window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data), "*"); }
        send("streamlit:componentReady", {apiVersion: 1});
        let initialized = false;
        window.addEventListener("message", function(event) {
            if (event.data.type === "streamlit:render" && !initialized) {
                initialized = true; const args = event.data.args;
                const canvas = document.getElementById('mycanvas');
                canvas.width = args.canvas_w; canvas.height = args.canvas_h;
                send("streamlit:setFrameHeight", {height: args.canvas_h + 100});
                const ctx = canvas.getContext('2d');
                const offscreen = document.createElement('canvas');
                offscreen.width = args.canvas_w; offscreen.height = args.canvas_h;
                const octx = offscreen.getContext('2d');
                const img = new Image();
                img.src = 'data:image/png;base64,' + args.img_b64;
                img.onload = () => ctx.drawImage(img, 0, 0);
                let painting = false;
                function draw(e) {
                    const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left; const y = e.clientY - r.top;
                    ctx.fillStyle = 'rgba(255,0,0,0.4)'; ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.fill();
                    octx.fillStyle = 'rgba(255,0,0,1.0)'; octx.beginPath(); octx.arc(x, y, 15, 0, Math.PI * 2); octx.fill();
                }
                canvas.addEventListener('mousedown', e => { painting = true; draw(e); });
                canvas.addEventListener('mousemove', e => { if (painting) draw(e); });
                canvas.addEventListener('mouseup', () => painting = false);
                canvas.addEventListener('mouseleave', () => painting = false);
                document.getElementById('btnClear').onclick = () => { ctx.drawImage(img, 0, 0); octx.clearRect(0, 0, offscreen.width, offscreen.height); };
                document.getElementById('btnSend').onclick = () => {
                    const maskB64 = offscreen.toDataURL('image/png').split(',')[1];
                    send("streamlit:setComponentValue", {value: maskB64});
                };
            }
        });
      </script>
    </body>
    </html>
    """)
st_mask_drawer = components.declare_component("mask_drawer", path=MASK_DIR)

# --- CORE UTILITIES ---
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

def unwarp_facade(image_pil, src_points, physical_width, physical_height, output_res=1024):
    image_cv = np.array(image_pil); aspect_ratio = physical_width / float(physical_height)
    out_w, out_h = (output_res, int(output_res / aspect_ratio)) if aspect_ratio > 1.0 else (int(output_res * aspect_ratio), output_res)
    src_pts = np.array([[p['x'], p['y']] for p in src_points], dtype="float32")
    dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return Image.fromarray(cv2.warpPerspective(image_cv, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC))

def process_depth_and_normals(image_pil, mask_b64, session, normal_strength):
    image_cv = np.array(image_pil); orig_h, orig_w = image_cv.shape[:2]
    img_resized = cv2.resize(image_cv, (518, 518), interpolation=cv2.INTER_CUBIC)
    img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0).astype(np.float32)
    depth_out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    depth_array = cv2.resize(np.squeeze(depth_out), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
    if mask_b64:
        mask_np = cv2.imdecode(np.frombuffer(base64.b64decode(mask_b64), np.uint8), cv2.IMREAD_UNCHANGED)
        if mask_np is not None:
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            binary_mask = np.where(mask_resized[:, :, 3] > 0, 255, 0).astype(np.uint8) if mask_resized.shape[2] == 4 else np.where(mask_resized[:, :, 2] > 0, 255, 0).astype(np.uint8)
            if np.any(binary_mask): depth_normalized = cv2.inpaint((depth_normalized * 255).astype(np.uint8), binary_mask, 3, cv2.INPAINT_NS).astype(np.float32) / 255.0
    depth_filtered = cv2.bilateralFilter(depth_normalized, d=9, sigmaColor=0.05, sigmaSpace=75)
    sobel_x, sobel_y = cv2.Sobel(depth_filtered, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(depth_filtered, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt((sobel_x * normal_strength)**2 + (sobel_y * normal_strength)**2 + 1.0)
    normal_map = np.stack([((-(sobel_x * normal_strength) / mag + 1.0) * 127.5), (((-(sobel_y * normal_strength) / mag) + 1.0) * 127.5), ((1.0 / mag) * 255)], axis=2).astype(np.uint8)
    return Image.fromarray((depth_filtered * 65535.0).astype(np.uint16), mode='I;16'), Image.fromarray(normal_map, mode='RGB')

def create_textured_plane(vertices, uvs, diffuse_img, normal_img, blank_rgba, face_name, project_id):
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if face_name == "Bot": mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[blank_rgba]*2); return mesh
    # Using PBRMaterial ensures the GLB carries the texture properly for the viewport
    mat = material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}", baseColorTexture=diffuse_img, normalTexture=normal_img, metallicFactor=0.0, roughnessFactor=0.9) if diffuse_img else material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}_Blank", baseColorFactor=blank_rgba)
    mesh.visual = texture.TextureVisuals(uv=uvs, material=mat); return mesh

# --- UI APP ---
st.set_page_config(page_title="MassingPro", layout="wide")
st.title("MassingPro")
faces = ["Front", "Back", "Left", "Right"]
for k in ['masks', 'warped']: 
    if k not in st.session_state: st.session_state[k] = {f: None for f in faces}

with st.sidebar:
    st.header("Project Dimensions")
    dim_x, dim_y, dim_z = st.number_input("Width (X) m", 1.0, 55.0, value=10.0), st.number_input("Depth (Y) m", 1.0, 55.0, value=15.0), st.number_input("Height (Z) m", 1.0, 68.0, value=12.0)
    blank_color, n_strength = st.color_picker("Empty Face Color", "#2A2D35"), st.slider("Normal Intensity", 0.1, 5.0, 2.0)

tabs = st.tabs([f" {f} Facade" for f in faces])
face_dims = {"Front": (dim_x, dim_z), "Back": (dim_x, dim_z), "Left": (dim_y, dim_z), "Right": (dim_y, dim_z)}
uv_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]]) 

for i, face in enumerate(faces):
    with tabs[i]:
        up_file = st.file_uploader(f"Upload {face}", type=["jpg", "png"], key=f"up_{face}")
        if up_file:
            raw = Image.open(up_file).convert("RGB")
            if st.session_state.warped[face] is None:
                cw, ch = min(800, raw.width), int(raw.height * (min(800, raw.width) / raw.width))
                pts_data = st_pts_picker(img_b64=pil_to_b64(raw.resize((cw, ch))), canvas_w=cw, canvas_h=ch, raw_w=raw.width, raw_h=raw.height, key=f"pts_{face}")
                if pts_data: st.session_state.warped[face] = unwarp_facade(raw, pts_data, *face_dims[face]); st.rerun()
            else:
                st.image(st.session_state.warped[face], use_container_width=True)
                if st.button("Reset Perspective", key=f"re_{face}"): st.session_state.warped[face] = None; st.rerun()
                cw, ch = min(800, st.session_state.warped[face].width), int(st.session_state.warped[face].height * (min(800, st.session_state.warped[face].width) / st.session_state.warped[face].width))
                mask_data = st_mask_drawer(img_b64=pil_to_b64(st.session_state.warped[face].resize((cw, ch))), canvas_w=cw, canvas_h=ch, key=f"mask_{face}")
                if mask_data: st.session_state.masks[face] = mask_data; st.success("✅ Mask Saved!")

if st.session_state.warped["Front"] and st.button("BUILD MASSING PRO ASSET", type="primary", use_container_width=True):
    with st.spinner("Compiling Visual GLB & Material Packages..."):
        session, project_id = ort.InferenceSession("depth_anything_v2_vits.onnx", providers=['CPUExecutionProvider']), str(random.randint(1000, 999999))
        blank_rgba, meshes, displacements, normals = [int(blank_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [255], [], {}, {}
        
        # Fixed Y-Up Geometry for correct Rhino/SketchUp orientation
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
                displacements[f], normals[f] = disp, norm
                meshes.append(create_textured_plane(plane_verts[f], uv_coords, st.session_state.warped[f], norm, blank_rgba, f, project_id))
            else: meshes.append(create_textured_plane(plane_verts[f], uv_coords, None, None, blank_rgba, f, project_id))
        meshes.extend([create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba, "Top", project_id), create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba, "Bot", project_id)])
        
        # --- EXPORT TO GLB FOR VIEWPORT VISIBILITY ---
        scene = trimesh.Scene(meshes)
        glb_data = scene.export(file_type='glb')
        
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write the self-contained GLB file
            zf.writestr(f"Geometry/MassingPro_{project_id}.glb", glb_data)
            
            # --- EXPORT MAPS AND ENSCAPE MATPKGS ---
            for f, img in displacements.items():
                m_name = f"MassingPro_{project_id}_{f}"
                a_f, d_f, n_f = f"{m_name}_Albedo.jpg", f"{m_name}_Displacement.png", f"{m_name}_Normal.png"
                ab, db, nb = io.BytesIO(), io.BytesIO(), io.BytesIO()
                st.session_state.warped[f].save(ab, format='JPEG'); img.save(db, format='PNG'); normals[f].save(nb, format='PNG')
                
                # Save raw maps as a backup
                zf.writestr(f"Maps/{a_f}", ab.getvalue()); zf.writestr(f"Maps/{d_f}", db.getvalue()); zf.writestr(f"Maps/{n_f}", nb.getvalue())
                
                # Build the Enscape material package sidecar
                mp_buf = io.BytesIO()
                with zipfile.ZipFile(mp_buf, "w", zipfile.ZIP_STORED) as mpz:
                    mpz.writestr(a_f, ab.getvalue()); mpz.writestr(d_f, db.getvalue()); mpz.writestr(n_f, nb.getvalue())
                    ens_json = {
                        "Version": 1, "Name": m_name, "Type": 0, "DoubleSided": False,
                        "DiffuseColor": [1.0, 1.0, 1.0],
                        "DiffuseTexture": {"File": a_f, "Transformation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "IsInverted": False, "Brightness": 1.0},
                        "BumpTexture": {"File": n_f, "Transformation": None, "IsInverted": False, "Brightness": 1.0},
                        "BumpAmount": 1.0, "BumpMapType": 3,
                        "Roughness": 0.9, "Metallic": 0.0, "Specular": 0.5,
                        "RoughnessTexture": {"File": d_f, "Transformation": None, "IsInverted": False, "Brightness": 1.0}
                    }
                    mpz.writestr("material.json", json.dumps(ens_json, indent=2))
                zf.writestr(f"Enscape_Ready/{m_name}.matpkg", mp_buf.getvalue())
        st.download_button("📦 DOWNLOAD HYBRID PACKAGE", buf.getvalue(), f"MassingPro_{project_id}.zip", "application/zip")
