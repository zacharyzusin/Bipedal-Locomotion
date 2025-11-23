#!/usr/bin/env python3
"""
Compute mass, COM, and inertia for a mesh in both CAD units (g, mm, g·mm²)
and MuJoCo units (kg, m, kg·m²).

Assumptions:
- Mesh geometry is in millimeters.
- Density is in g/cm³ (e.g. PLA ~ 1.2–1.3 g/cm³).
"""

import argparse
import numpy as np
import trimesh


def compute_mass_properties(mesh_path: str, density_g_per_cm3: float):
    # Load mesh (in mm)
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight; volume and inertia may be inaccurate.")

    # Volume in mm³
    volume_mm3 = mesh.volume
    if volume_mm3 < 0:
        print("Warning: negative volume; flipping sign.")
        volume_mm3 = -volume_mm3
    volume_cm3 = volume_mm3 / 1000.0

    # ------------------------------------------------------------
    # Work in meters for mass / inertia
    # ------------------------------------------------------------
    mesh_m = mesh.copy()
    mesh_m.apply_scale(0.001)  # mm → m

    # Density: g/cm³ → kg/m³
    density_kg_per_m3 = density_g_per_cm3 * 1000.0

    # Volume in m³ (after scaling)
    volume_m3 = mesh_m.volume
    mass_kg = density_kg_per_m3 * volume_m3

    # Center of mass in meters
    com_m = np.array(mesh_m.center_mass)

    # Inertia about COM, for density = 1 kg/m³
    # According to trimesh docs, moment_inertia is at center_mass
    inertia_unit = np.array(mesh_m.moment_inertia)  # kg·m² for density=1

    # Scale inertia to actual density
    inertia_kgm2 = inertia_unit * density_kg_per_m3

    # ------------------------------------------------------------
    # Convert to CAD-ish view (g, mm, g·mm²)
    # ------------------------------------------------------------
    mass_g = mass_kg * 1000.0
    com_mm = com_m * 1000.0
    # 1 kg·m² = 10^7 g·mm²
    inertia_gmm2 = inertia_kgm2 * 1e7

    # Ensure symmetry & positive-definite (for MuJoCo)
    inertia_kgm2 = 0.5 * (inertia_kgm2 + inertia_kgm2.T)
    eigvals, _ = np.linalg.eigh(inertia_kgm2)
    min_eig = eigvals.min()
    if min_eig <= 0:
        eps = 1e-8 - min_eig
        print(f"Warning: inertia not positive-definite, shifting by {eps:.3e}.")
        inertia_kgm2 += np.eye(3) * eps

    return {
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_cm3,
        "mass_g": mass_g,
        "mass_kg": mass_kg,
        "com_mm": com_mm,
        "com_m": com_m,
        "inertia_gmm2": inertia_gmm2,
        "inertia_kgm2": inertia_kgm2,
    }


def format_fullinertia_for_mujoco(I: np.ndarray) -> str:
    """
    Format a 3x3 inertia tensor as a MuJoCo fullinertia string:
    [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """
    Ixx = I[0, 0]
    Iyy = I[1, 1]
    Izz = I[2, 2]
    Ixy = I[0, 1]
    Ixz = I[0, 2]
    Iyz = I[1, 2]
    vals = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    return " ".join(f"{v:.8e}" for v in vals)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mass properties (mass, COM, inertia) of a mesh "
                    "and print them in g/mm and kg/m units plus a MuJoCo <inertial> snippet."
    )
    parser.add_argument("mesh", help="Path to the mesh file (STL/OBJ/etc.)")
    parser.add_argument(
        "--density",
        type=float,
        default=1.25,
        help="Material density in g/cm³ (default: 1.25 for PLA-ish)",
    )
    parser.add_argument(
        "--part-name",
        type=str,
        default="part",
        help="Name of the part/body (only used for labeling).",
    )

    args = parser.parse_args()
    props = compute_mass_properties(args.mesh, args.density)

    mass_g = props["mass_g"]
    mass_kg = props["mass_kg"]
    com_mm = props["com_mm"]
    com_m = props["com_m"]
    inertia_gmm2 = props["inertia_gmm2"]
    inertia_kgm2 = props["inertia_kgm2"]

    print("\n===== CAD-STYLE UNITS =====")
    print(f"Density: {args.density:.4f} g/cm³")
    print(f"Volume:  {props['volume_mm3']:.3f} mm³  ({props['volume_cm3']:.3f} cm³)")
    print(f"Mass:    {mass_g:.6f} g")
    print("COM (mm):", com_mm)
    print("Inertia (g·mm²):")
    print(inertia_gmm2)

    print("\n===== MUJOCO-READY UNITS =====")
    print(f"Mass:    {mass_kg:.6f} kg")
    print("COM (m):", com_m)
    print("Inertia (kg·m²):")
    print(inertia_kgm2)
    print("============================\n")

    # MuJoCo snippet
    full = format_fullinertia_for_mujoco(inertia_kgm2)
    print("MuJoCo <inertial> snippet (paste inside <body>):\n")
    print(f'<inertial pos="{com_m[0]:.8f} {com_m[1]:.8f} {com_m[2]:.8f}" '
          f'mass="{mass_kg:.8f}" fullinertia="{full}"/>')


if __name__ == "__main__":
    main()