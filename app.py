if st.session_state.warped["Front"] and st.button("BUILD MASSING PRO ASSET", type="primary"):
    with st.spinner("Processing AI Maps & Packaging OBJ..."):
        # Load model and generate Project ID
        session = load_onnx_model()
        project_id = str(random.randint(1000, 999999))
        
        blank_rgba = [int(blank_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [255]
        meshes, displacements, normals = [], {}, {}
        
        # Y-UP ORIENTATION MATRICES
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
        
        meshes.append(create_textured_plane(plane_verts["Top"], uv_coords, None, None, blank_rgba, "Top", project_id))
        meshes.append(create_textured_plane(plane_verts["Bot"], uv_coords, None, None, blank_rgba, "Bot", project_id))
        
        # --- EXPORT TO ZIP ---
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # --- FIXED OBJ EXPORT LOGIC ---
            # Trimesh returns a dictionary when exporting a scene as OBJ
            export_data = trimesh.Scene(meshes).export(file_type='obj')
            
            if isinstance(export_data, dict):
                for filename, data in export_data.items():
                    # Trimesh uses 'material.mtl' by default; we rename to match project
                    final_name = filename if "material" not in filename else f"MassingPro_{project_id}.mtl"
                    # Ensure data is bytes for the zip write
                    content = data.encode('utf-8') if isinstance(data, str) else data
                    zf.writestr(f"Geometry/{final_name}", content)
            else:
                zf.writestr(f"Geometry/MassingPro_{project_id}.obj", export_data)
            
            # --- MAPS & ENCSAPE READY FOLDERS ---
            for f, img in displacements.items():
                m_name = f"MassingPro_{project_id}_{f}"
                a_f, d_f, n_f = f"{m_name}_Albedo.jpg", f"{m_name}_Displacement.png", f"{m_name}_Normal.png"
                
                ab, db, nb = io.BytesIO(), io.BytesIO(), io.BytesIO()
                st.session_state.warped[f].save(ab, format='JPEG')
                img.save(db, format='PNG')
                normals[f].save(nb, format='PNG')
                
                # Raw Map Storage
                zf.writestr(f"Maps/{a_f}", ab.getvalue())
                zf.writestr(f"Maps/{d_f}", db.getvalue())
                zf.writestr(f"Maps/{n_f}", nb.getvalue())
                
                # Enscape .matpkg Generation (Using your verified JSON schema)
                mp_buf = io.BytesIO()
                with zipfile.ZipFile(mp_buf, "w", zipfile.ZIP_STORED) as mpz:
                    mpz.writestr(a_f, ab.getvalue())
                    mpz.writestr(d_f, db.getvalue())
                    mpz.writestr(n_f, nb.getvalue())
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

        st.success(f"✅ Asset Built! Project ID: {project_id}")
        st.download_button("📦 DOWNLOAD PRO PACKAGE", buf.getvalue(), f"MassingPro_{project_id}.zip", "application/zip")
