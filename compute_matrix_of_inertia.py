import numpy as np
import trimesh

def mass_properties_from_obj(obj_path: str, density: float = 1.0):
    """
    Compute center of mass and inertia tensor for a closed mesh with
    homogeneous density.
    """

    mesh = trimesh.load(obj_path, force='mesh')

    # If multiple submeshes, merge them
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight; volume and inertia may be inaccurate.")

    # This is a *property*, not a function on your version of trimesh
    mp = mesh.mass_properties  # <-- FIXED

    # Extract values for density = 1
    volume = mp["volume"]
    com = np.array(mp["center_mass"])
    inertia_com_1 = np.array(mp["inertia"])  # inertia around COM

    # Scale by density
    mass = density * volume
    inertia_com = density * inertia_com_1

    return com, mass, inertia_com


if __name__ == "__main__":
    obj_file = "meshes/obj/hips.obj"
    density = 1.24  # example density (match your units)

    com, mass, inertia = mass_properties_from_obj(obj_file, density)

    print("Mass:", mass)
    print("COM:", com)
    print("Inertia tensor about COM:\n", inertia)
