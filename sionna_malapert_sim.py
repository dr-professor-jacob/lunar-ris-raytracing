"""
Malapert Massif RIS Raytracing Simulation
Calculates radio coverage for Artemis III using Sionna-RT.
"""

import os
import sqlite3
import numpy as np
import trimesh
import plotly.graph_objects as go

#### SIONNA / TENSORFLOW SETUP ####
os.environ["DRJIT_LIBLLVM_PATH"] = r"C:\Program Files\LLVM\bin\LLVM-C.dll"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt import PathSolver, RadioMaterial

#### MISSION PARAMETERS ####
# HLS Base Station height (Starship)
STARSHIP_HLS_ANTENNA_HEIGHT_M = 50.0

# RIS Relay mast height (LUNARSABER)
LUNARSABER_MAST_HEIGHT_M = 100.0


def clear_sqlite():
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
    print(f"Exporting {freq_ghz} GHz telemetry data to MCP SQLite database...")
    db_path = (
        r"C:\Users\jrick\Desktop\Antigravity_DropBox\busy-raman\lunar_telemetry.db"
    )

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

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

        # Insert new telemetry
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

        cursor.executemany(
            """
            INSERT INTO malapert_coverage (frequency_ghz, x_coord, y_coord, z_coord, baseline_power_dbm, ris_assisted_power_dbm)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            rows,
        )

        conn.commit()
        conn.close()
        print(f"Database export successful for {freq_ghz} GHz!")
    except Exception as e:
        print(f"Failed to export to database: {e}")


def compute_frequency(freq_hz, scene, solver, bs_pos, ris_pos, rx_positions, mesh):
    print(f"\n#### Computing Ray Tracing for {freq_hz/1e9:.1f} GHz ####")

    scene.frequency = freq_hz
    paths = solver(scene=scene, max_depth=1, samples_per_src=2000000)
    a_list, tau_list = paths.cir()

    # Extract path coefficients

    #### BASE STATION POWER (TX 0) ####
    a_bs_real = a_list[0][:, :, 0:1, ...]
    a_bs_imag = a_list[1][:, :, 0:1, ...]

    p_bs = tf.reduce_sum(
        tf.square(a_bs_real) + tf.square(a_bs_imag), axis=[1, 2, 3, 4, 5]
    )
    power_bs = 10 * np.log10(np.maximum(p_bs.numpy(), 1e-20)) + 43

    #### RIS PROXY POWER (TX 1) ####
    a_ris_real = a_list[0][:, :, 1:2, ...]
    a_ris_imag = a_list[1][:, :, 1:2, ...]

    p_ris = tf.reduce_sum(
        tf.square(a_ris_real) + tf.square(a_ris_imag), axis=[1, 2, 3, 4, 5]
    )

    ris_base_power = 36.0
    power_ris = 10 * np.log10(np.maximum(p_ris.numpy(), 1e-20)) + ris_base_power

    # Rover connects to strongest signal
    power_combined = np.maximum(power_bs, power_ris)
    return power_bs, power_combined


def main():
    print(f"Initializing NVIDIA Sionna-RT v{sionna.__version__}...")

    # 1. LOAD TERRAIN MESH
    mesh = trimesh.load("models/terrain/nasa_malapert_dem.obj", force="mesh")

    # 80x80 grid for Rovers
    x = np.linspace(-2500, 4500, 80)
    y = np.linspace(-2000, 2000, 80)
    xx, yy = np.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    ray_origins = np.column_stack((xx_flat, yy_flat, np.full_like(xx_flat, 8000)))
    ray_directions = np.tile([0, 0, -1], (len(xx_flat), 1))

    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )

    rx_positions = locations.copy()
    rx_positions[:, 2] += 1.5

    # 2. LOAD SIONNA SCENE
    scene = load_scene("models/terrain/malapert_scene.xml")
    regolith_mat = RadioMaterial(
        "lunar_regolith", relative_permittivity=3.1, conductivity=0.001
    )
    scene.add(regolith_mat)
    scene.get("merged-shapes").radio_material = "lunar_regolith"
    scene.synthetic_array = True

    # Locate BS on Earth-facing slope (X = -500)
    bs_loc, _, _ = mesh.ray.intersects_location([[-500, 0, 8000]], [[0, 0, -1]])

    # Locate RIS on Massif Peak (X = 0)
    ris_loc, _, _ = mesh.ray.intersects_location([[0, 0, 8000]], [[0, 0, -1]])

    bs_pos = np.array(
        [bs_loc[0][0], bs_loc[0][1], bs_loc[0][2] + STARSHIP_HLS_ANTENNA_HEIGHT_M]
    )

    # Apply mast height to clear convex terrain
    ris_pos = np.array(
        [ris_loc[0][0], ris_loc[0][1], ris_loc[0][2] + LUNARSABER_MAST_HEIGHT_M]
    )

    print(f"\n[Validation] Base Station at: {bs_pos}")
    print(f"[Validation] RIS at: {ris_pos}")

    #### BASE STATION SETUP ####
    tx_bs = Transmitter(name="BS_Lander", position=bs_pos)
    # Set antenna pattern
    try:
        scene.tx_array = PlanarArray(
            num_rows=8, num_cols=8, pattern="tr38901", polarization="V"
        )
    except Exception:
        scene.tx_array = PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V"
        )
    scene.add(tx_bs)

    #### RIS PROXY SETUP ####
    tx_ris = Transmitter(name="RIS_Relay", position=ris_pos)
    scene.add(tx_ris)
    tx_ris.array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    # Add Rovers to scene
    for i, pos in enumerate(rx_positions):
        rx = Receiver(name=f"Rover_{i}", position=pos)
        scene.add(rx)

    # 3. COMPUTE THE PHYSICS FOR MULTIPLE FREQUENCIES
    solver = PathSolver()
    frequencies_ghz = [3.5, 10.0, 28.0]

    clear_sqlite()

    all_results = {}

    for freq in frequencies_ghz:
        p_bs, p_comb = compute_frequency(
            freq * 1e9, scene, solver, bs_pos, ris_pos, rx_positions, mesh
        )

        # 4. EXPORT DATA
        export_to_sqlite(rx_positions, p_bs, p_comb, freq)
        all_results[f"power_bs_{freq}"] = p_bs
        all_results[f"power_combined_{freq}"] = p_comb

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
    print("Loading Malapert simulation data...")
    data = np.load("results/malapert_coverage_data.npz")
    rx_positions = data["rx_positions"]
    bs_pos = data["bs_pos"]
    ris_pos = data["ris_pos"]

    power_bs = data["power_bs_28.0"]
    power_combined = data["power_combined_28.0"]

    print("Loading Malapert terrain mesh...")
    mesh = trimesh.load("models/terrain/nasa_malapert_dem.obj", force="mesh")
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vertices[:, 2],
            colorscale="Greys",
            opacity=0.9,
            name="Lunar Terrain",
            hoverinfo="skip",
            showscale=False,
            flatshading=True,
        )
    )

    val_b = power_bs > -190
    fig.add_trace(
        go.Scatter3d(
            x=rx_positions[val_b, 0],
            y=rx_positions[val_b, 1],
            z=rx_positions[val_b, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=power_bs[val_b],
                colorscale="Turbo",
                cmin=-130,
                cmax=-60,
                opacity=0.8,
                colorbar=dict(title="Signal (dBm)", x=0.8),
            ),
            name="Baseline (No RIS)",
            hovertemplate="Power: %{marker.color:.1f} dBm<extra></extra>",
            visible=True,
            showlegend=False,
        )
    )

    val_c = power_combined > -190
    fig.add_trace(
        go.Scatter3d(
            x=rx_positions[val_c, 0],
            y=rx_positions[val_c, 1],
            z=rx_positions[val_c, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=power_combined[val_c],
                colorscale="Turbo",
                cmin=-130,
                cmax=-60,
                opacity=0.8,
                colorbar=dict(title="Signal (dBm)", x=0.8),
            ),
            name="RIS Assisted",
            hovertemplate="Power: %{marker.color:.1f} dBm<extra></extra>",
            visible=False,
            showlegend=False,
        )
    )

    hls_ground_z = bs_pos[2] - 50.0
    hls_cylinder = create_cylinder(
        radius=4.5,
        height=50.0,
        center_x=bs_pos[0],
        center_y=bs_pos[1],
        base_z=hls_ground_z,
        color="silver",
    )
    fig.add_trace(hls_cylinder)

    ris_ground_z = ris_pos[2] - 100.0
    mast_cylinder = create_cylinder(
        radius=1.5,
        height=100.0,
        center_x=ris_pos[0],
        center_y=ris_pos[1],
        base_z=ris_ground_z,
        color="darkgray",
    )
    fig.add_trace(mast_cylinder)

    ris_panel = create_box(
        center_x=ris_pos[0],
        center_y=ris_pos[1],
        center_z=ris_pos[2],
        length=5,
        width=5,
        height=0.5,
        color="cyan",
    )
    fig.add_trace(ris_panel)

    np.random.seed(42)
    valley_mask = (rx_positions[:, 0] > 1000) & (power_combined > -110)
    valley_rx = rx_positions[valley_mask]

    static_annotations = [
        dict(
            x=bs_pos[0],
            y=bs_pos[1],
            z=bs_pos[2] + 200,
            text="Starship HLS",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="red",
            ax=-60,
            ay=-50,
        ),
        dict(
            x=ris_pos[0],
            y=ris_pos[1],
            z=ris_pos[2] + 200,
            text="100m Mast (RIS)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="cyan",
            ax=60,
            ay=-50,
        ),
    ]

    if len(valley_rx) > 0:
        sorted_indices = np.argsort(valley_rx[:, 1])
        if len(sorted_indices) >= 3:
            rover_indices = [
                sorted_indices[len(sorted_indices) // 8],
                sorted_indices[len(sorted_indices) // 2],
                sorted_indices[-(len(sorted_indices) // 8)],
            ]
        else:
            rover_indices = sorted_indices[:3]
        rover_locs = valley_rx[rover_indices]
        rover_rssi_base = power_bs[valley_mask][rover_indices]
        rover_rssi_comb = power_combined[valley_mask][rover_indices]

        for i, loc in enumerate(rover_locs):
            rover_box = create_box(
                center_x=loc[0],
                center_y=loc[1],
                center_z=loc[2] + 18.5,
                length=60,
                width=40,
                height=40,
                color="gold",
            )
            fig.add_trace(rover_box)

        text_base = [
            f"<b>LTV Rover</b><br>({rssi:.1f} dBm)" for rssi in rover_rssi_base
        ]
        z_stagger = np.array([400, 800, 1200])[: len(rover_locs)]
        fig.add_trace(
            go.Scatter3d(
                x=rover_locs[:, 0],
                y=rover_locs[:, 1],
                z=rover_locs[:, 2] + z_stagger,
                mode="text",
                text=text_base,
                textfont=dict(color="white", size=11),
                hoverinfo="skip",
                showlegend=False,
                name="Rover Text Baseline",
            )
        )

        text_comb = [
            f"<b>LTV Rover</b><br>({rssi:.1f} dBm)" for rssi in rover_rssi_comb
        ]
        fig.add_trace(
            go.Scatter3d(
                x=rover_locs[:, 0],
                y=rover_locs[:, 1],
                z=rover_locs[:, 2] + z_stagger,
                mode="text",
                text=text_comb,
                textfont=dict(color="white", size=11),
                hoverinfo="skip",
                showlegend=False,
                name="Rover Text RIS",
                visible=False,
            )
        )

    vis_baseline = []
    vis_ris = []
    for trace in fig.data:
        if trace.name in ["RIS Assisted", "Rover Text RIS"]:
            vis_baseline.append(False)
            vis_ris.append(True)
        elif trace.name in ["Baseline (No RIS)", "Rover Text Baseline"]:
            vis_baseline.append(True)
            vis_ris.append(False)
        else:
            vis_baseline.append(True)
            vis_ris.append(True)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                buttons=list(
                    [
                        dict(
                            label="Baseline Coverage (No RIS)",
                            method="update",
                            args=[{"visible": vis_baseline}],
                        ),
                        dict(
                            label="RIS Assisted Coverage (100m Mast)",
                            method="update",
                            args=[{"visible": vis_ris}],
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                font=dict(color="black"),
            ),
        ],
        title="<b>Artemis III: Malapert Massif 28 GHz Physical Simulation</b><br><sup>Featuring SpaceX Starship HLS (50m) and LUNARSABER Relay (100m)</sup>",
        title_x=0.5,
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5)),
            annotations=static_annotations,
        ),
        margin=dict(l=0, r=0, b=0, t=100),
    )

    output_file = "results/dashboard.html"
    fig.write_html(output_file)
    print(f"Successfully generated {output_file}!")


if __name__ == "__main__":
    main()
