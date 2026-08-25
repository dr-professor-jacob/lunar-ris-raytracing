"""
Malapert Massif RIS Raytracing Simulation
Calculates radio coverage for Artemis III using Sionna-RT.
"""

import os
import sqlite3
import numpy as np
import trimesh
import plotly.graph_objects as go

# Set Windows path for LLVM so Sionna works.
os.environ["DRJIT_LIBLLVM_PATH"] = r"C:\Program Files\LLVM\bin\LLVM-C.dll"
# Hide TensorFlow spam.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt import PathSolver, RadioMaterial

# Starship antenna is 50 meters high.
STARSHIP_HLS_ANTENNA_HEIGHT_M = 50.0

# LUNARSABER mast is 100 meters high.
LUNARSABER_MAST_HEIGHT_M = 100.0


def clear_sqlite():
    # Delete old database table.
    db_path = (
        r"C:\Users\jrick\Desktop\Antigravity_DropBox\busy-raman\lunar_telemetry.db"
    )
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS malapert_coverage")
        conn.commit()
        conn.close()
    except Exception:
        pass


def export_to_sqlite(rx_positions, power_bs, power_combined, freq_ghz):
    # Save data to database.
    print(f"Exporting {freq_ghz} GHz telemetry data to MCP SQLite database...")
    db_path = (
        r"C:\Users\jrick\Desktop\Antigravity_DropBox\busy-raman\lunar_telemetry.db"
    )

    try:
        # Connect to DB.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Make new table.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS malapert_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frequency_ghz REAL,
                x_coord REAL,
                y_coord REAL,
                z_coord REAL,
                baseline_power_dbm REAL,
                ris_assisted_power_dbm REAL
            )
        """)

        # Package row data.
        rows = []
        for i in range(len(rx_positions)):
            rows.append(
                (
                    float(freq_ghz),
                    float(rx_positions[i][0]),
                    float(rx_positions[i][1]),
                    float(rx_positions[i][2]),
                    float(power_bs[i]),
                    float(power_combined[i]),
                )
            )

        # Insert rows.
        cursor.executemany(
            """
            INSERT INTO malapert_coverage (frequency_ghz, x_coord, y_coord, z_coord, baseline_power_dbm, ris_assisted_power_dbm)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            rows,
        )

        # Save and close.
        conn.commit()
        conn.close()
        print(f"Database export successful for {freq_ghz} GHz!")
    except Exception as e:
        print(f"Failed to export to database: {e}")


def compute_frequency(freq_hz, scene, solver, bs_pos, ris_pos, rx_positions, mesh):
    # Calculate radio physics.
    print(f"\n#### Computing Ray Tracing for {freq_hz/1e9:.1f} GHz ####")

    # Set frequency.
    scene.frequency = freq_hz
    # Shoot rays. Max 1 bounce.
    paths = solver(scene=scene, max_depth=1, samples_per_src=2000000)
    # Get complex path coefficients.
    a_list, tau_list = paths.cir()

    # Get Base Station math (TX 0).
    a_bs_real = a_list[0][:, :, 0:1, ...]
    a_bs_imag = a_list[1][:, :, 0:1, ...]

    # Square math to get Watts.
    p_bs = tf.reduce_sum(
        tf.square(a_bs_real) + tf.square(a_bs_imag), axis=[1, 2, 3, 4, 5]
    )
    # Convert Watts to dBm.
    power_bs = 10 * np.log10(np.maximum(p_bs.numpy(), 1e-20)) + 43

    # Get RIS math (TX 1).
    a_ris_real = a_list[0][:, :, 1:2, ...]
    a_ris_imag = a_list[1][:, :, 1:2, ...]

    # Square math to get Watts.
    p_ris = tf.reduce_sum(
        tf.square(a_ris_real) + tf.square(a_ris_imag), axis=[1, 2, 3, 4, 5]
    )

    # Convert Watts to dBm.
    ris_base_power = 36.0
    power_ris = 10 * np.log10(np.maximum(p_ris.numpy(), 1e-20)) + ris_base_power

    # Rover connects to strongest signal.
    power_combined = np.maximum(power_bs, power_ris)
    return power_bs, power_combined


def main():
    print(f"Initializing NVIDIA Sionna-RT v{sionna.__version__}...")

    # Load 3D mountain model.
    mesh = trimesh.load("models/terrain/nasa_malapert_dem.obj", force="mesh")

    # Make 80x80 grid for Rovers.
    x = np.linspace(-2500, 4500, 80)
    y = np.linspace(-2000, 2000, 80)
    xx, yy = np.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Set rays high above grid.
    ray_origins = np.column_stack((xx_flat, yy_flat, np.full_like(xx_flat, 8000)))
    # Point rays straight down.
    ray_directions = np.tile([0, 0, -1], (len(xx_flat), 1))

    # Shoot rays to hit mountain surface.
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )

    # Put rovers slightly above ground.
    rx_positions = locations.copy()
    rx_positions[:, 2] += 1.5

    # Load Sionna XML scene.
    scene = load_scene("models/terrain/malapert_scene.xml")
    # Add Moon dirt physics (Regolith).
    regolith_mat = RadioMaterial(
        "lunar_regolith", relative_permittivity=3.1, conductivity=0.001
    )
    scene.add(regolith_mat)
    # Paint mountain with moon dirt.
    scene.get("merged-shapes").radio_material = "lunar_regolith"
    scene.synthetic_array = True

    # Find spot for Base Station.
    bs_loc, _, _ = mesh.ray.intersects_location([[-500, 0, 8000]], [[0, 0, -1]])

    # Find spot for RIS Mast.
    ris_loc, _, _ = mesh.ray.intersects_location([[0, 0, 8000]], [[0, 0, -1]])

    # Add 50m Starship height.
    bs_pos = np.array(
        [bs_loc[0][0], bs_loc[0][1], bs_loc[0][2] + STARSHIP_HLS_ANTENNA_HEIGHT_M]
    )

    # Add 100m Mast height.
    ris_pos = np.array(
        [ris_loc[0][0], ris_loc[0][1], ris_loc[0][2] + LUNARSABER_MAST_HEIGHT_M]
    )

    print(f"\n[Validation] Base Station at: {bs_pos}")
    print(f"[Validation] RIS at: {ris_pos}")

    # Make Base Station Transmitter.
    tx_bs = Transmitter(name="BS_Lander", position=bs_pos)
    # Set antenna type.
    scene.tx_array = PlanarArray(
        num_rows=8, num_cols=8, pattern="tr38901", polarization="V"
    )
    scene.add(tx_bs)

    # Make RIS Proxy Transmitter.
    tx_ris = Transmitter(name="RIS_Relay", position=ris_pos)
    scene.add(tx_ris)
    tx_ris.array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    # Make Receiver antenna type.
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    # Add Rovers to physics engine.
    for i, pos in enumerate(rx_positions):
        rx = Receiver(name=f"Rover_{i}", position=pos)
        scene.add(rx)

    # Make raytracer object.
    solver = PathSolver()
    # Set frequency list.
    frequencies_ghz = [3.5, 10.0, 28.0]

    # Clean old database.
    clear_sqlite()

    all_results = {}

    # Loop over frequencies.
    for freq in frequencies_ghz:
        # Run physics math.
        p_bs, p_comb = compute_frequency(
            freq * 1e9, scene, solver, bs_pos, ris_pos, rx_positions, mesh
        )

        # Save data.
        export_to_sqlite(rx_positions, p_bs, p_comb, freq)
        all_results[f"power_bs_{freq}"] = p_bs
        all_results[f"power_combined_{freq}"] = p_comb

    # Save to NPZ file.
    np.savez(
        "results/malapert_coverage_data.npz",
        rx_positions=rx_positions,
        bs_pos=bs_pos,
        ris_pos=ris_pos,
        **all_results,
    )
    print("\nSimulation Complete. Data exported to results/malapert_coverage_data.npz")

    print("\nBuilding 3D Interactive Dashboard...")
    build_dashboard()


def create_cylinder(radius, height, center_x, center_y, base_z, color, resolution=16):
    # Make a 3D cylinder shape.
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta) + center_x
    y = radius * np.sin(theta) + center_y
    z_bottom = np.full(resolution, base_z)
    z_top = np.full(resolution, base_z + height)

    x_grid = np.array([x, x])
    y_grid = np.array([y, y])
    z_grid = np.array([z_bottom, z_top])

    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
    )


def create_box(center_x, center_y, center_z, length, width, height, color):
    # Make a 3D box shape.
    dx = length / 2
    dy = width / 2
    dz = height / 2

    x = [
        center_x - dx,
        center_x + dx,
        center_x + dx,
        center_x - dx,
        center_x - dx,
        center_x + dx,
        center_x + dx,
        center_x - dx,
    ]
    y = [
        center_y - dy,
        center_y - dy,
        center_y + dy,
        center_y + dy,
        center_y - dy,
        center_y - dy,
        center_y + dy,
        center_y + dy,
        center_y - dy,
    ]
    z = [
        center_z - dz,
        center_z - dz,
        center_z - dz,
        center_z - dz,
        center_z + dz,
        center_z + dz,
        center_z + dz,
        center_z + dz,
    ]

    return go.Mesh3d(
        x=x, y=y, z=z, alphahull=0, color=color, flatshading=True, hoverinfo="skip"
    )


def build_dashboard():
    import numpy as np
    import trimesh
    import plotly.graph_objects as go

    print("Loading Malapert simulation data...")
    data = np.load("results/malapert_coverage_data.npz")
    rx_positions = data["rx_positions"]
    bs_pos = data["bs_pos"]
    ris_pos = data["ris_pos"]

    print("Loading Malapert terrain mesh...")
    mesh = trimesh.load("models/terrain/nasa_malapert_dem.obj", force="mesh")
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure()

    # Trace 0: Terrain
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="#2a2a2a", opacity=0.95, name="Lunar Terrain", hoverinfo="skip", showscale=False, flatshading=True, lighting=dict(ambient=0.4, diffuse=0.6, specular=0.1, roughness=0.9)))

    # Trace 1, 2, 3: Infrastructure
    fig.add_trace(create_cylinder(4.5, 50.0, bs_pos[0], bs_pos[1], bs_pos[2] - 50.0, "silver"))
    fig.add_trace(create_cylinder(1.5, 100.0, ris_pos[0], ris_pos[1], ris_pos[2] - 100.0, "darkgray"))
    fig.add_trace(create_box(ris_pos[0], ris_pos[1], ris_pos[2], 5, 5, 0.5, "cyan"))

    power_combined_28 = data["power_combined_28.0"]
    np.random.seed(42)
    valley_mask = (rx_positions[:, 0] > 1000) & (power_combined_28 > -110)
    valley_rx = rx_positions[valley_mask]
    
    rover_locs = []
    rover_indices = []
    if len(valley_rx) > 0:
        sorted_indices = np.argsort(valley_rx[:, 1])
        if len(sorted_indices) >= 3:
            rover_indices = [sorted_indices[len(sorted_indices) // 8], sorted_indices[len(sorted_indices) // 2], sorted_indices[-(len(sorted_indices) // 8)]]
        else:
            rover_indices = sorted_indices[:3]
        rover_locs = valley_rx[rover_indices]

        for loc in rover_locs:
            fig.add_trace(create_box(loc[0], loc[1], loc[2] + 18.5, 60, 40, 40, "gold"))

    static_annotations = [
        dict(x=bs_pos[0], y=bs_pos[1], z=bs_pos[2] + 200, text="Starship HLS", showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="red", ax=-60, ay=-50),
        dict(x=ris_pos[0], y=ris_pos[1], z=ris_pos[2] + 200, text="100m Mast (RIS)", showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="cyan", ax=60, ay=-50),
    ]

    frequencies = [3.5, 10.0, 28.0]
    for f in frequencies:
        power_bs = data[f"power_bs_{f}"]
        power_combined = data[f"power_combined_{f}"]
        
        val_b = power_bs > -190
        fig.add_trace(go.Scatter3d(x=rx_positions[val_b, 0], y=rx_positions[val_b, 1], z=rx_positions[val_b, 2], mode="markers", marker=dict(size=6, color=power_bs[val_b], colorscale="Turbo", cmin=-130, cmax=-60, opacity=0.8, colorbar=dict(title="Signal (dBm)", x=0.8)), name=f"Baseline (No RIS) - {f} GHz", hovertemplate="Power: %{marker.color:.1f} dBm<extra></extra>", visible=(f == 28.0), showlegend=False))
        
        val_c = power_combined > -190
        fig.add_trace(go.Scatter3d(x=rx_positions[val_c, 0], y=rx_positions[val_c, 1], z=rx_positions[val_c, 2], mode="markers", marker=dict(size=6, color=power_combined[val_c], colorscale="Turbo", cmin=-130, cmax=-60, opacity=0.8, colorbar=dict(title="Signal (dBm)", x=0.8)), name=f"RIS Assisted - {f} GHz", hovertemplate="Power: %{marker.color:.1f} dBm<extra></extra>", visible=False, showlegend=False))

        if len(rover_locs) > 0:
            rover_rssi_base = power_bs[valley_mask][rover_indices]
            rover_rssi_comb = power_combined[valley_mask][rover_indices]

            text_base = [f"<b>LTV Rover</b><br>({rssi:.1f} dBm)" for rssi in rover_rssi_base]
            z_stagger = np.array([400, 800, 1200])[: len(rover_locs)]
            fig.add_trace(go.Scatter3d(x=rover_locs[:, 0], y=rover_locs[:, 1], z=rover_locs[:, 2] + z_stagger, mode="text", text=text_base, textfont=dict(color="white", size=11), hoverinfo="skip", showlegend=False, name=f"Rover Text Baseline - {f} GHz", visible=(f == 28.0)))
            
            text_comb = [f"<b>LTV Rover</b><br>({rssi:.1f} dBm)" for rssi in rover_rssi_comb]
            fig.add_trace(go.Scatter3d(x=rover_locs[:, 0], y=rover_locs[:, 1], z=rover_locs[:, 2] + z_stagger, mode="text", text=text_comb, textfont=dict(color="white", size=11), hoverinfo="skip", showlegend=False, name=f"Rover Text RIS - {f} GHz", visible=False))

    buttons = []
    for target_name in ["Baseline", "RIS"]:
        for f in frequencies:
            vis_array = []
            for trace in fig.data:
                if not getattr(trace, "name", None) or "GHz" not in trace.name:
                    vis_array.append(True)
                else:
                    if f"{f} GHz" in trace.name:
                        if target_name == "Baseline" and "Baseline" in trace.name:
                            vis_array.append(True)
                        elif target_name == "RIS" and "RIS" in trace.name:
                            vis_array.append(True)
                        else:
                            vis_array.append(False)
                    else:
                        vis_array.append(False)
            
            label = f"{f} GHz - {'Baseline (No RIS)' if target_name == 'Baseline' else 'RIS Assisted (100m Mast)'}"
            buttons.append(dict(label=label, method="update", args=[{"visible": vis_array}]))

    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="down", x=0.01, y=0.99, xanchor="left", yanchor="top", buttons=buttons, pad={"r": 10, "t": 10}, showactive=True, font=dict(color="black"))],
        title="<b>Artemis III: Malapert Massif Multi-Band Physical Simulation</b><br><sup>Featuring SpaceX Starship HLS (50m) and LUNARSABER Relay (100m)</sup>",
        title_x=0.5, template="plotly_dark",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation (m)", aspectmode="data", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5)), annotations=static_annotations),
        margin=dict(l=0, r=0, b=0, t=100)
    )

    output_file = "results/dashboard.html"
    fig.write_html(output_file)
    print(f"Successfully generated {output_file}!")


if __name__ == "__main__":
    main()
