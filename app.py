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
          <p style="margin:0 0 8px 0;font-size:13px;color:#d1d5db;">Click the <strong>four corners</strong> of the building face in order: top-left &rarr; top-right &rarr; bottom-right &rarr; bottom-left. Drag any point to fine-tune.</p>
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
            <span style="font-size:11px;color:#9ca3af;">Zoom:</span>
            <button id="zoomOut" style="padding:0 6px;background:#1f2937;color:white;border:1px solid #374151;border-radius:3px;cursor:pointer;font-size:11px;line-height:1.8;">&#8722;</button>
            <span id="zoomLabel" style="font-size:11px;color:#d1d5db;min-width:24px;text-align:center;">1&times;</span>
            <button id="zoomIn" style="padding:0 6px;background:#1f2937;color:white;border:1px solid #374151;border-radius:3px;cursor:pointer;font-size:11px;line-height:1.8;">&#43;</button>
          </div>
          <canvas id="mycanvas" style="cursor:crosshair;display:block;"></canvas><br>
          <button id="btnClear" style="margin-top:8px;padding:8px 16px;background:#333;color:white;border:none;border-radius:4px;cursor:pointer;margin-right:8px;">Clear</button>
          <button id="btnSend" style="margin-top:8px;padding:8px 16px;background:#ff4b4b;color:white;border:none;border-radius:4px;cursor:pointer;">Extract Perspective</button>
      </div>
      <script>
        function send(type, data) { window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data), "*"); }
        send("streamlit:componentReady", {apiVersion: 1});
        let initialized = false; let points = []; let dragging = -1;
        let scaleX = 1; let scaleY = 1; let drawState = null;
        let zoom = 1; let baseW = 0; let baseH = 0;
        const ZOOM_STEPS = [1, 1.5, 2, 3];
        window.addEventListener("message", function(event) {
            if (event.data.type !== "streamlit:render") return;
            const args = event.data.args;
            if (!initialized) {
                initialized = true;
                const canvas = document.getElementById('mycanvas');
                baseW = args.canvas_w; baseH = args.canvas_h;
                canvas.width = baseW; canvas.height = baseH;
                send("streamlit:setFrameHeight", {height: baseH + 110});
                const ctx = canvas.getContext('2d'); const img = new Image();
                scaleX = args.raw_w / baseW; scaleY = args.raw_h / baseH;
                img.src = 'data:image/png;base64,' + args.img_b64;
                function setZoom(newZoom) {
                    const ratio = newZoom / zoom; zoom = newZoom;
                    points = points.map(p => ({x: p.x * ratio, y: p.y * ratio}));
                    canvas.width = Math.round(baseW * zoom);
                    canvas.height = Math.round(baseH * zoom);
                    scaleX = args.raw_w / canvas.width;
                    scaleY = args.raw_h / canvas.height;
                    document.getElementById('zoomLabel').textContent = zoom + '\u00d7';
                    send("streamlit:setFrameHeight", {height: canvas.height + 110});
                    if (img.complete) drawState();
                }
                document.getElementById('zoomOut').onclick = () => {
                    const i = ZOOM_STEPS.indexOf(zoom);
                    if (i > 0) setZoom(ZOOM_STEPS[i - 1]);
                };
                document.getElementById('zoomIn').onclick = () => {
                    const i = ZOOM_STEPS.indexOf(zoom);
                    if (i < ZOOM_STEPS.length - 1) setZoom(ZOOM_STEPS[i + 1]);
                };
                drawState = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height); ctx.lineWidth = 3;
                    const labels = ['TL', 'TR', 'BR', 'BL'];
                    for (let i=0; i<points.length; i++) {
                        const px = points[i].x, py = points[i].y;
                        const fill = dragging === i ? '#ffaa00' : '#ff4b4b';
                        ctx.beginPath(); ctx.arc(px, py, 10, 0, Math.PI*2);
                        ctx.fillStyle = fill; ctx.fill();
                        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.stroke();
                        ctx.font = 'bold 11px sans-serif'; ctx.fillStyle = '#ffffff';
                        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                        ctx.fillText(String(i + 1), px, py);
                        if (labels[i]) {
                            ctx.font = '11px sans-serif'; ctx.fillStyle = fill;
                            ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                            ctx.fillText(labels[i], px + 13, py - 6);
                        }
                    }
                    if (points.length > 1) {
                        ctx.strokeStyle = '#ff4b4b'; ctx.lineWidth = 2;
                        ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
                        for (let i=1; i<points.length; i++) ctx.lineTo(points[i].x, points[i].y);
                        if (points.length === 4) ctx.lineTo(points[0].x, points[0].y);
                        ctx.stroke();
                    }
                };
                function hitTest(x, y) {
                    for (let i=0; i<points.length; i++) {
                        const dx = points[i].x - x, dy = points[i].y - y;
                        if (Math.sqrt(dx*dx + dy*dy) < 12) return i;
                    }
                    return -1;
                }
                img.onload = () => { drawState(); };
                canvas.addEventListener('mousedown', e => {
                    const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left; const y = e.clientY - r.top;
                    const hit = hitTest(x, y);
                    if (hit >= 0) { dragging = hit; }
                    else if (points.length < 4) { points.push({x, y}); drawState(); }
                });
                canvas.addEventListener('mousemove', e => {
                    if (dragging < 0) return;
                    const r = canvas.getBoundingClientRect();
                    points[dragging] = {x: e.clientX - r.left, y: e.clientY - r.top};
                    drawState();
                });
                canvas.addEventListener('mouseup', () => { dragging = -1; });
                canvas.addEventListener('mouseleave', () => { dragging = -1; });
                document.getElementById('btnClear').onclick = () => { points = []; drawState(); };
                document.getElementById('btnSend').onclick = () => {
                    if (points.length !== 4) return alert('Please select exactly 4 points.');
                    const finalPts = points.map(p => ({x: p.x * scaleX, y: p.y * scaleY}));
                    send("streamlit:setComponentValue", {value: finalPts});
                };
            }
            // Always apply incoming initial_pts (e.g. after Auto-Detect re-render)
            if (args.initial_pts && args.initial_pts.length === 4 && points.length === 0) {
                points = args.initial_pts.map(p => ({x: p.x / scaleX, y: p.y / scaleY}));
                if (drawState) drawState();
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
          <p style="margin:0 0 8px 0;font-size:13px;color:#d1d5db;">🖌️ Brush over anything that isn&#39;t part of the building surface — sky, scaffolding, vehicles, trees. This keeps the 3D texture clean. Click <strong>Save Mask</strong> when done.</p>
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
            <span style="font-size:11px;color:#9ca3af;">Zoom:</span>
            <button id="zoomOut" style="padding:0 6px;background:#1f2937;color:white;border:1px solid #374151;border-radius:3px;cursor:pointer;font-size:11px;line-height:1.8;">&#8722;</button>
            <span id="zoomLabel" style="font-size:11px;color:#d1d5db;min-width:24px;text-align:center;">1&times;</span>
            <button id="zoomIn" style="padding:0 6px;background:#1f2937;color:white;border:1px solid #374151;border-radius:3px;cursor:pointer;font-size:11px;line-height:1.8;">&#43;</button>
          </div>
          <canvas id="mycanvas" style="cursor:crosshair;display:block;"></canvas><br>
          <button id="btnClear" style="margin-top:8px;padding:8px 16px;background:#333;color:white;border:none;border-radius:4px;cursor:pointer;margin-right:8px;">Clear Brush</button>
          <button id="btnSend" style="margin-top:8px;padding:8px 16px;background:#ff4b4b;color:white;border:none;border-radius:4px;cursor:pointer;">Save Mask</button>
      </div>
      <script>
        function send(type, data) { window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data), "*"); }
        send("streamlit:componentReady", {apiVersion: 1});
        let initialized = false;
        let zoom = 1; let baseW = 0; let baseH = 0;
        const ZOOM_STEPS = [1, 1.5, 2, 3];
        window.addEventListener("message", function(event) {
            if (event.data.type === "streamlit:render" && !initialized) {
                initialized = true; const args = event.data.args;
                const canvas = document.getElementById('mycanvas');
                baseW = args.canvas_w; baseH = args.canvas_h;
                canvas.width = baseW; canvas.height = baseH;
                send("streamlit:setFrameHeight", {height: baseH + 110});
                const ctx = canvas.getContext('2d');
                const offscreen = document.createElement('canvas');
                offscreen.width = baseW; offscreen.height = baseH;
                const octx = offscreen.getContext('2d');
                const img = new Image();
                img.src = 'data:image/png;base64,' + args.img_b64;
                function redrawVisible() {
                    canvas.width = Math.round(baseW * zoom);
                    canvas.height = Math.round(baseH * zoom);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 0.4;
                    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 1.0;
                }
                img.onload = () => redrawVisible();
                function setZoom(newZoom) {
                    zoom = newZoom;
                    document.getElementById('zoomLabel').textContent = zoom + '\u00d7';
                    send("streamlit:setFrameHeight", {height: Math.round(baseH * zoom) + 110});
                    redrawVisible();
                }
                document.getElementById('zoomOut').onclick = () => {
                    const i = ZOOM_STEPS.indexOf(zoom); if (i > 0) setZoom(ZOOM_STEPS[i - 1]);
                };
                document.getElementById('zoomIn').onclick = () => {
                    const i = ZOOM_STEPS.indexOf(zoom); if (i < ZOOM_STEPS.length - 1) setZoom(ZOOM_STEPS[i + 1]);
                };
                let painting = false;
                function draw(e) {
                    const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left; const y = e.clientY - r.top;
                    const ox = x / zoom; const oy = y / zoom;
                    octx.fillStyle = 'rgba(255,0,0,1.0)'; octx.beginPath(); octx.arc(ox, oy, 15, 0, Math.PI * 2); octx.fill();
                    redrawVisible();
                }
                canvas.addEventListener('mousedown', e => { painting = true; draw(e); });
                canvas.addEventListener('mousemove', e => { if (painting) draw(e); });
                canvas.addEventListener('mouseup', () => painting = false);
                canvas.addEventListener('mouseleave', () => painting = false);
                document.getElementById('btnClear').onclick = () => { octx.clearRect(0, 0, offscreen.width, offscreen.height); redrawVisible(); };
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

PREVIEW_DIR = "preview_frontend"
os.makedirs(PREVIEW_DIR, exist_ok=True)
with open(f"{PREVIEW_DIR}/index.html", "w") as f:
    f.write("""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:transparent;overflow:hidden;">
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body { font-family: sans-serif; }
    #panel {
      width: 100%; background: #111827;
      border: 1px solid #374151; border-radius: 10px;
      overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    #toolbar {
      height: 36px; background: #0f172a;
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 12px; cursor: pointer; user-select: none;
    }
    #title { color: #9ca3af; font-size: 11px; font-weight: 600; letter-spacing: .06em; }
    #toggleBtn { background: none; border: none; color: #6b7280; font-size: 16px; cursor: pointer; }
    model-viewer { width: 100%; background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%); }
    #hint { padding: 5px 12px; background: #0f172a; color: #4b5563; font-size: 10px; }
  </style>
  <div id="panel">
    <div id="toolbar">
      <span id="title">3D MODEL PREVIEW</span>
      <button id="toggleBtn" data-expanded="false">&#x2922; Expand</button>
    </div>
    <model-viewer id="mv"
      camera-controls auto-rotate
      auto-rotate-delay="800" rotation-per-second="15deg"
      shadow-intensity="0.5" exposure="1.1" tone-mapping="neutral"
      style="height:308px;">
    </model-viewer>
    <div id="hint">Drag to orbit &middot; Scroll to zoom &middot; Right-click drag to pan</div>
  </div>
  <script>
    function send(type, data) {
      window.parent.postMessage(Object.assign({isStreamlitMessage:true, type}, data), "*");
    }
    send("streamlit:componentReady", {apiVersion: 1});

    let currentGlb = null;
    let isExpanded = false;

    const btn = document.getElementById('toggleBtn');
    const mv = document.getElementById('mv');

    function applyExpanded(val) {
      isExpanded = val;
      const h = isExpanded ? 650 : 370;
      btn.dataset.expanded = String(isExpanded);
      btn.innerHTML = isExpanded ? '&#x2921; Collapse' : '&#x2922; Expand';
      mv.style.height = (h - 62) + 'px';
      send("streamlit:setFrameHeight", {height: h});
    }

    document.getElementById('toolbar').addEventListener('click', () => {
      applyExpanded(!isExpanded);
    });

    window.addEventListener("message", function(e) {
      if (e.data.type !== "streamlit:render") return;
      const args = e.data.args;
      if (args.glb_b64 !== currentGlb) {
        currentGlb = args.glb_b64;
        applyExpanded(true);
        mv.setAttribute('src', 'data:model/gltf-binary;base64,' + args.glb_b64);
      } else {
        applyExpanded(isExpanded);
      }
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    });
  </script>
</body>
</html>""")
st_preview_panel = components.declare_component("preview_panel", path=PREVIEW_DIR)

@st.cache_resource
def _load_depth_session():
    return ort.InferenceSession("depth_anything_v2_vits.onnx", providers=['CPUExecutionProvider'])

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

def _glb_add_doublesided(glb_bytes: bytes) -> bytes:
    """Patch GLB materials to doubleSided=True for web preview only. Export is unaffected."""
    try:
        json_len = int.from_bytes(glb_bytes[12:16], 'little')
        gltf = json.loads(glb_bytes[20:20 + json_len].decode('utf-8'))
        for mat in gltf.get('materials', []):
            mat['doubleSided'] = True
        new_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        pad = (4 - len(new_json) % 4) % 4
        new_json += b' ' * pad
        rest = glb_bytes[20 + json_len:]
        new_total = 12 + 8 + len(new_json) + len(rest)
        return (
            (0x46546C67).to_bytes(4, 'little') +
            (2).to_bytes(4, 'little') +
            new_total.to_bytes(4, 'little') +
            len(new_json).to_bytes(4, 'little') +
            (0x4E4F534A).to_bytes(4, 'little') +
            new_json + rest
        )
    except Exception:
        return glb_bytes

def create_textured_plane(vertices, uvs, diffuse_img, normal_img, blank_rgba, face_name, project_id):
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if face_name == "Bot": mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[blank_rgba]*2); return mesh
    mat = material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}", baseColorTexture=diffuse_img, normalTexture=normal_img, metallicFactor=0.0, roughnessFactor=0.9) if diffuse_img else material.PBRMaterial(name=f"MassingPro_{project_id}_{face_name}_Blank", baseColorFactor=blank_rgba)
    mesh.visual = texture.TextureVisuals(uv=uvs, material=mat); return mesh

def generate_collada_dae(project_id, plane_verts, img_buffers, blank_rgba):
    """Generate a Collada 1.4.1 XML string for SketchUp import with textures."""
    all_faces = ["Front", "Back", "Left", "Right", "Top", "Bot"]
    textured = set(img_buffers.keys())
    br, bg, bb = blank_rgba[0]/255.0, blank_rgba[1]/255.0, blank_rgba[2]/255.0

    # --- library_images ---
    images_xml = ""
    for f in all_faces:
        if f in textured:
            m_name, a_f, d_f, n_f = img_buffers[f][0], img_buffers[f][1], img_buffers[f][2], img_buffers[f][3]
            images_xml += f'''    <image id="img-{f}-albedo" name="{m_name}_Albedo">
      <init_from>{a_f}</init_from>
    </image>
    <image id="img-{f}-normal" name="{m_name}_Normal">
      <init_from>{n_f}</init_from>
    </image>
'''

    # --- library_effects ---
    effects_xml = ""
    for f in all_faces:
        if f in textured:
            m_name = img_buffers[f][0]
            n_f = img_buffers[f][3]
            effects_xml += f'''    <effect id="effect-{f}">
      <profile_COMMON>
        <newparam sid="{f}-albedo-surface"><surface type="2D"><init_from>img-{f}-albedo</init_from></surface></newparam>
        <newparam sid="{f}-albedo-sampler"><sampler2D><source>{f}-albedo-surface</source></sampler2D></newparam>
        <newparam sid="{f}-normal-surface"><surface type="2D"><init_from>img-{f}-normal</init_from></surface></newparam>
        <newparam sid="{f}-normal-sampler"><sampler2D><source>{f}-normal-surface</source></sampler2D></newparam>
        <technique sid="common">
          <phong>
            <ambient><color>1 1 1 1</color></ambient>
            <diffuse><texture texture="{f}-albedo-sampler" texcoord="UVMap"/></diffuse>
            <specular><color>0 0 0 1</color></specular>
            <shininess><float>0</float></shininess>
          </phong>
        </technique>
        <extra><technique profile="FCOLLADA">
          <bump bumptype="NORMALMAP"><texture texture="{f}-normal-sampler" texcoord="UVMap"/></bump>
        </technique></extra>
      </profile_COMMON>
    </effect>
'''
        else:
            effects_xml += f'''    <effect id="effect-{f}">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse><color>{br:.4f} {bg:.4f} {bb:.4f} 1</color></diffuse>
            <specular><color>0 0 0 1</color></specular>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
'''

    # --- library_materials ---
    materials_xml = ""
    for f in all_faces:
        if f in textured:
            m_name = img_buffers[f][0]
        else:
            m_name = f"MassingPro_{project_id}_{f}_Blank"
        materials_xml += f'''    <material id="mat-{f}" name="{m_name}">
      <instance_effect url="#effect-{f}"/>
    </material>
'''

    # --- library_geometries ---
    # Front and Back face cross products produce inward normals with the default
    # winding order — reverse their triangles so normals point outward.
    inward_faces = {"Front", "Back"}
    geometries_xml = ""
    for f in all_faces:
        verts = plane_verts[f]
        pos_vals = " ".join(f"{v:.6f}" for vert in verts for v in vert)
        # UVs: same as existing exports [[0,1],[1,1],[1,0],[0,0]]
        uv_vals = "0 1  1 1  1 0  0 0"
        # Reversed winding [0,2,1],[0,3,2] for faces whose default order yields an inward normal
        p_str = "0 0 2 2 1 1  0 0 3 3 2 2" if f in inward_faces else "0 0 1 1 2 2  0 0 2 2 3 3"
        geometries_xml += f'''    <geometry id="geo-{f}" name="{f}">
      <mesh>
        <source id="geo-{f}-pos">
          <float_array id="geo-{f}-pos-arr" count="12">{pos_vals}</float_array>
          <technique_common>
            <accessor source="#geo-{f}-pos-arr" count="4" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="geo-{f}-uvs">
          <float_array id="geo-{f}-uvs-arr" count="8">{uv_vals}</float_array>
          <technique_common>
            <accessor source="#geo-{f}-uvs-arr" count="4" stride="2">
              <param name="S" type="float"/><param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="geo-{f}-verts">
          <input semantic="POSITION" source="#geo-{f}-pos"/>
        </vertices>
        <triangles count="2" material="mat-{f}-symbol">
          <input semantic="VERTEX" source="#geo-{f}-verts" offset="0"/>
          <input semantic="TEXCOORD" source="#geo-{f}-uvs" offset="1" set="0"/>
          <p>{p_str}</p>
        </triangles>
      </mesh>
    </geometry>
'''

    # --- library_visual_scenes ---
    nodes_xml = ""
    for f in all_faces:
        nodes_xml += f'''      <node id="node-{f}" name="{f}" type="NODE">
        <instance_geometry url="#geo-{f}">
          <bind_material>
            <technique_common>
              <instance_material symbol="mat-{f}-symbol" target="#mat-{f}">
                <bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/>
              </instance_material>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
'''

    return f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>MassingPro</authoring_tool></contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_images>
{images_xml}  </library_images>
  <library_effects>
{effects_xml}  </library_effects>
  <library_materials>
{materials_xml}  </library_materials>
  <library_geometries>
{geometries_xml}  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
{nodes_xml}    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
'''

# --- AUTO PERSPECTIVE DETECTION ---
def order_quad_points(pts):
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).flatten()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])

def detect_facade_corners(image_pil):
    """Detect the 4 facade corners using edge + contour analysis.
    Returns list of {x, y} in raw image coordinates, ordered TL→TR→BR→BL.
    Falls back to a centred 85% crop if no clear quad is found."""
    img_cv = np.array(image_pil.convert("RGB"))
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad, best_area = None, 0
    min_area = w * h * 0.08
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.04, 0.06, 0.10]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area and area > min_area:
                    best_area = area
                    best_quad = approx.reshape(4, 2).astype(float)
                break
    if best_quad is not None:
        pts = order_quad_points(best_quad)
    else:
        mx, my = w * 0.075, h * 0.075
        pts = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]])
    return [{"x": float(p[0]), "y": float(p[1])} for p in pts]

# --- RHINO SCRIPT GENERATOR ---
def generate_rhino_script(project_id):
    return f'''"""
MassingPro Auto-Material Script
Project ID : {project_id}
Instructions: Place this file in the same folder as the exported OBJ and
              texture images. Open the OBJ in Rhino, then run this script
              via Tools > Python Script > Run.
"""
import os
import scriptcontext as sc
import Rhino

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    doc = sc.doc
    applied = 0

    for mat in doc.Materials:
        if not mat.Name.startswith("MassingPro_{project_id}_"):
            continue

        albedo = os.path.join(script_dir, mat.Name + "_Albedo.jpg")
        normal = os.path.join(script_dir, mat.Name + "_Normal.png")

        if os.path.exists(albedo):
            mat.SetBitmapTexture(albedo)

        if os.path.exists(normal):
            mat.SetBumpTexture(normal)

        mat.CommitChanges()
        applied += 1

    doc.Views.Redraw()
    print("MassingPro: Applied textures to {{}} material(s).".format(applied))

run()
'''

# --- UI APP ---
st.set_page_config(page_title="MassingPro", page_icon="🏗️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, .stButton, input, label, .stTextInput,
.stNumberInput, .stSelectbox, .stRadio, .stSlider, .stFileUploader, .stTabs, .stExpander {
    font-family: 'Neutra Text', 'Neutra Display', 'Jost', 'Futura', sans-serif !important;
}
/* Number input: no border, compact +/- buttons */
[data-testid="stNumberInput"] input {
    border: none !important;
}
[data-testid="stNumberInput"] button {
    border: none !important;
    min-width: 20px !important;
    padding: 0 2px !important;
    font-size: 11px !important;
}
/* Tighten sidebar vertical spacing so all items fit without scrolling */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)
faces = ["Front", "Back", "Left", "Right"]
for k in ['masks', 'warped', 'auto_pts']:
    if k not in st.session_state: st.session_state[k] = {f: None for f in faces}
if 'preview_glb_b64' not in st.session_state: st.session_state['preview_glb_b64'] = None
if 'pkg_zip' not in st.session_state: st.session_state['pkg_zip'] = None
if 'pkg_name' not in st.session_state: st.session_state['pkg_name'] = None

with st.sidebar:
    st.markdown('<p style="font-size:2rem;font-weight:700;letter-spacing:-0.02em;margin:0 0 10px 0;line-height:1.1;">MassingPro</p>', unsafe_allow_html=True)
    st.caption("Generate a textured 3D massing model from facade photos.")
    st.divider()
    st.subheader("Building Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        dim_x = st.number_input("Width (X) m", 1.0, 55.0, value=10.0, step=0.25)
        dim_z = st.number_input("Height (Z) m", 1.0, 68.0, value=12.0, step=0.25)
    with col2:
        dim_y = st.number_input("Depth (Y) m", 1.0, 55.0, value=15.0, step=0.25)
    st.divider()
    st.subheader("Appearance")
    blank_color = st.color_picker("Untextured Face Colour", "#2A2D35")
    n_strength = st.slider(
        "Surface Relief",
        0.1, 5.0, 2.0,
        help="How strongly the facade depth is translated into surface texture detail. Higher = bolder relief. 2.0 is a good default for most facades."
    )
    st.divider()
    with st.expander("How to use MassingPro"):
        st.markdown(
            """
**1. Set building dimensions** — width, depth, and height in metres.

**2. Upload a facade photo** in the Front tab (required). Back, Left, and Right are optional.

**3. Frame the facade** — click the four corners of the building face in order, then click **Extract Perspective**. Use **Auto-Detect** as a starting point.

**4. Remove unwanted elements** (optional) — brush over sky, scaffolding, vehicles, or trees.

**5. Choose export format** and click **Build 3D Model**. Review the preview, then download.
            """
        )

tabs = st.tabs([f"{f} Facade" for f in faces])

face_dims = {"Front": (dim_x, dim_z), "Back": (dim_x, dim_z), "Left": (dim_y, dim_z), "Right": (dim_y, dim_z)}
uv_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

for i, face in enumerate(faces):
    with tabs[i]:
        up_file = st.file_uploader(f"Upload {face}", type=["jpg", "png"], key=f"up_{face}")
        if up_file:
            raw = Image.open(up_file).convert("RGB")
            if st.session_state.warped[face] is None:
                cw = min(800, raw.width)
                ch = int(raw.height * (cw / raw.width))

                col_a, col_b = st.columns([1, 5])
                with col_a:
                    if st.button("🎯 Auto-Detect", key=f"auto_{face}"):
                        st.session_state.auto_pts[face] = detect_facade_corners(raw)
                with col_b:
                    if st.session_state.auto_pts[face]:
                        st.caption("✅ Corners auto-detected — drag to refine, or Clear and click manually.")
                    else:
                        st.caption("Optional shortcut. You can also click the 4 corners directly on the canvas.")

                pts_data = st_pts_picker(
                    img_b64=pil_to_b64(raw.resize((cw, ch))),
                    canvas_w=cw, canvas_h=ch,
                    raw_w=raw.width, raw_h=raw.height,
                    initial_pts=st.session_state.auto_pts[face],
                    key=f"pts_{face}"
                )
                if pts_data:
                    st.session_state.warped[face] = unwarp_facade(raw, pts_data, *face_dims[face])
                    st.session_state.auto_pts[face] = None
                    st.rerun()
            else:
                col_preview, _ = st.columns([3, 2])
                with col_preview:
                    st.image(st.session_state.warped[face], use_container_width=True)
                if st.button("Reset Perspective", key=f"re_{face}"):
                    st.session_state.warped[face] = None
                    st.session_state.auto_pts[face] = None
                    st.session_state['preview_glb_b64'] = None
                    st.session_state['pkg_zip'] = None
                    st.session_state['pkg_name'] = None
                    st.rerun()
                cw = min(800, st.session_state.warped[face].width)
                ch = int(st.session_state.warped[face].height * (cw / st.session_state.warped[face].width))
                mask_data = st_mask_drawer(img_b64=pil_to_b64(st.session_state.warped[face].resize((cw, ch))), canvas_w=cw, canvas_h=ch, key=f"mask_{face}")
                if mask_data: st.session_state.masks[face] = mask_data; st.success("✅ Mask Saved!")

st.divider()

# --- USER FORMAT SELECTION ---
if st.session_state.warped["Front"]:
    st.markdown("### Export")
    export_format = st.radio(
        "Export format:",
        [
            "OBJ + MTL (.obj) — for Enscape material replacement in Rhino",
            "GLTF Binary (.glb) — for Rhino & SketchUp viewports",
            "Collada (.dae) — for SketchUp with V-Ray or Enscape"
        ],
        horizontal=False
    )

    if st.button("Build 3D Model", type="primary", use_container_width=True):
        with st.spinner("Compiling Package..."):
            session, project_id = _load_depth_session(), str(random.randint(1000, 999999))
            blank_rgba, meshes, displacements, normals = [int(blank_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [255], [], {}, {}

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
                else:
                    meshes.append(create_textured_plane(plane_verts[f], uv_coords, None, None, blank_rgba, f, project_id))
            meshes.extend([
                create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba, "Top", project_id),
                create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba, "Bot", project_id)
            ])

            scene = trimesh.Scene(meshes)

            # --- GENERATE PREVIEW GLB (double-sided via reversed winding) ---
            try:
                ds_meshes = []
                for m in meshes:
                    ds_meshes.append(m)
                    flipped = m.copy()
                    flipped.faces = flipped.faces[:, ::-1]
                    ds_meshes.append(flipped)
                preview_glb_bytes = trimesh.Scene(ds_meshes).export(file_type='glb')
                st.session_state['preview_glb_b64'] = base64.b64encode(preview_glb_bytes).decode()
            except Exception:
                st.session_state['preview_glb_b64'] = None

            # --- PRE-RENDER IMAGE BUFFERS ---
            img_buffers = {}
            for f, disp_img in displacements.items():
                m_name = f"MassingPro_{project_id}_{f}"
                a_f = f"{m_name}_Albedo.jpg"
                d_f = f"{m_name}_Displacement.png"
                n_f = f"{m_name}_Normal.png"
                ab, db, nb = io.BytesIO(), io.BytesIO(), io.BytesIO()
                st.session_state.warped[f].save(ab, format='JPEG', quality=95, subsampling=0)
                disp_img.save(db, format='PNG')
                normals[f].save(nb, format='PNG')
                img_buffers[f] = (m_name, a_f, d_f, n_f, ab.getvalue(), db.getvalue(), nb.getvalue())

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

                # --- GEOMETRY EXPORT (flat in ZIP root) ---
                if "glb" in export_format.lower():
                    zf.writestr(f"MassingPro_{project_id}.glb", scene.export(file_type='glb'))
                elif "dae" in export_format.lower():
                    dae_xml = generate_collada_dae(project_id, plane_verts, img_buffers, blank_rgba)
                    zf.writestr(f"MassingPro_{project_id}.dae", dae_xml)
                    for f, (m_name, a_f, d_f, n_f, ab_bytes, db_bytes, nb_bytes) in img_buffers.items():
                        zf.writestr(a_f, ab_bytes)
                        zf.writestr(n_f, nb_bytes)
                else:
                    obj_name = f"MassingPro_{project_id}.obj"
                    mtl_name = f"MassingPro_{project_id}.mtl"
                    export_data = scene.export(file_type='obj')

                    # Write OBJ
                    if isinstance(export_data, dict):
                        for fn, d in export_data.items():
                            if fn.endswith('.obj'):
                                text = d if isinstance(d, str) else d.decode('utf-8', errors='replace')
                                for old_fn in export_data.keys():
                                    if old_fn.endswith('.mtl'):
                                        text = text.replace(f"mtllib {old_fn}", f"mtllib {mtl_name}")
                                zf.writestr(obj_name, text.encode('utf-8'))
                    elif isinstance(export_data, (str, bytes)):
                        text = export_data if isinstance(export_data, str) else export_data.decode('utf-8', errors='replace')
                        text = text.replace("mtllib material.mtl", f"mtllib {mtl_name}")
                        zf.writestr(obj_name, text.encode('utf-8'))

                    # Write MTL with map_Kd + map_bump (Enscape reads both natively)
                    face_maps = {data[0]: (data[1], data[3]) for data in img_buffers.values()}
                    mtl_lines = []
                    for mesh in meshes:
                        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                            mat = mesh.visual.material
                            mat_name = getattr(mat, 'name', None)
                            if mat_name:
                                entry = ["Ka 1.0 1.0 1.0", "Kd 0.9 0.9 0.9", "Ks 0.0 0.0 0.0", "d 1.0", "illum 2"]
                                if mat_name in face_maps:
                                    a_f, n_f = face_maps[mat_name]
                                    entry.append(f"map_Kd {a_f}")
                                    entry.append(f"map_bump -bm 1.0 {n_f}")
                                mtl_lines += [f"newmtl {mat_name}"] + entry + [""]
                    if mtl_lines:
                        zf.writestr(mtl_name, "\n".join(mtl_lines))

                    # Albedo + Normal flat alongside OBJ
                    for f, (m_name, a_f, d_f, n_f, ab_bytes, db_bytes, nb_bytes) in img_buffers.items():
                        zf.writestr(a_f, ab_bytes)
                        zf.writestr(n_f, nb_bytes)

                    # Rhino auto-material script
                    zf.writestr(f"MassingPro_{project_id}_ApplyMaterials.py", generate_rhino_script(project_id))

                # --- DISPLACEMENT MAPS + MATPKG (flat in ZIP root) ---
                for f, (m_name, a_f, d_f, n_f, ab_bytes, db_bytes, nb_bytes) in img_buffers.items():
                    zf.writestr(d_f, db_bytes)

                    mp_buf = io.BytesIO()
                    with zipfile.ZipFile(mp_buf, "w", zipfile.ZIP_STORED) as mpz:
                        mpz.writestr(a_f, ab_bytes)
                        mpz.writestr(d_f, db_bytes)
                        mpz.writestr(n_f, nb_bytes)
                        ens_json = {
                            "Version": 1, "Name": m_name, "Type": 0, "DoubleSided": False,
                            "DiffuseColor": [1.0, 1.0, 1.0],
                            "UseColorChannel": True,
                            "DiffuseTexture": {"File": a_f, "Transformation": None, "IsInverted": False, "Brightness": 1.0},
                            "ImageFade": 1.0, "Opacity": 1.0, "MaskTexture": None,
                            "TintColor": [1.0, 1.0, 1.0], "IsSolidGlass": False,
                            "EmissiveColor": [1.0, 1.0, 1.0], "EmissiveStrength": 0.0,
                            "BumpTexture": {"File": n_f, "Transformation": None, "IsInverted": False, "Brightness": 1.0},
                            "BumpAmount": 1.0, "BumpMapType": 3, "Roughness": 0.9,
                            "RoughnessTexture": {"File": d_f, "Transformation": None, "IsInverted": False, "Brightness": 1.0},
                            "Metallic": 0.0, "Specular": 0.5, "Refraction": 1.5
                        }
                        mpz.writestr("material.json", json.dumps(ens_json, indent=2))
                    zf.writestr(f"{m_name}.matpkg", mp_buf.getvalue())

            st.session_state['pkg_zip'] = buf.getvalue()
            st.session_state['pkg_name'] = f"MassingPro_{project_id}.zip"

    if st.session_state.get('preview_glb_b64'):
        st_preview_panel(
            glb_b64=st.session_state['preview_glb_b64'],
            panel_height=370,
            height=370,
            key="preview_panel"
        )

    if st.session_state.get('pkg_zip'):
        st.download_button("⬇️ Download Package", st.session_state['pkg_zip'], st.session_state['pkg_name'], "application/zip", use_container_width=True)
