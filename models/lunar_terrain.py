"""
3D Lunar Terrain & Crater Geometry Generator for Ray Tracing
"""

import os
import numpy as np

class LunarTerrainGenerator:
    """
    Generates 3D elevation maps and OBJ meshes representing lunar south pole crater topology.
    """
    def __init__(self, radius_m=1000.0, depth_m=250.0, rim_height_m=50.0, resolution=50):
        self.radius = radius_m
        self.depth = depth_m
        self.rim_height = rim_height_m
        self.resolution = resolution
        
    def elevation_func(self, x, y):
        """
        Analytical elevation function Z(x,y) modeling a lunar crater:
        - Flat crater bottom at Z = -depth_m for r < 0.4 * radius
        - Parabolic / Gaussian transition up to rim at r = radius (Z = +rim_height)
        - Exponential decay outside rim for r > radius
        """
        r = np.sqrt(x**2 + y**2)
        r_flat = 0.35 * self.radius
        
        z = np.zeros_like(r)
        
        # Inside crater floor & inner slopes
        mask_inside = r <= self.radius
        # Interpolate between -depth and +rim_height
        norm_r = np.clip((r[mask_inside] - r_flat) / (self.radius - r_flat), 0.0, 1.0)
        z[mask_inside] = -self.depth + (self.depth + self.rim_height) * (norm_r**2)
        
        # Deepest floor smoothing
        mask_floor = r < r_flat
        z[mask_floor] = -self.depth
        
        # Exterior apron slope outside rim
        mask_outside = r > self.radius
        z[mask_outside] = self.rim_height * np.exp(- (r[mask_outside] - self.radius) / (0.4 * self.radius))
        
        return z

    def generate_mesh_grid(self, bounds=1500.0):
        """
        Creates 2D meshgrid arrays (X, Y, Z) for the terrain surface.
        """
        lin = np.linspace(-bounds, bounds, self.resolution)
        X, Y = np.meshgrid(lin, lin)
        Z = self.elevation_func(X, Y)
        return X, Y, Z

    def export_obj(self, filepath, bounds=1500.0):
        """
        Exports the 3D terrain as a standard Wavefront .OBJ file for Sionna-RT / Mitsuba.
        """
        X, Y, Z = self.generate_mesh_grid(bounds=bounds)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        rows, cols = X.shape
        vertices = []
        for i in range(rows):
            for j in range(cols):
                vertices.append((X[i, j], Y[i, j], Z[i, j]))
                
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Vertex 1-based indexing for OBJ
                v1 = i * cols + j + 1
                v2 = i * cols + (j + 1) + 1
                v3 = (i + 1) * cols + j + 1
                v4 = (i + 1) * cols + (j + 1) + 1
                
                # Two triangles per grid square
                faces.append((v1, v2, v3))
                faces.append((v3, v2, v4))
                
        with open(filepath, 'w') as f:
            f.write("# Lunar South Pole Crater Mesh generated for Sionna-RT\n")
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
                
        print(f"[TerrainGenerator] Exported 3D lunar terrain mesh to: {filepath} ({len(vertices)} vertices, {len(faces)} faces)")
        return filepath
