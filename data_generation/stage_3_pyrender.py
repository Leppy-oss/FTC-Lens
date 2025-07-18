import os
import json
import shutil
import timeit
import trimesh
import pyrender
import numpy as np
from PIL import Image
from random import sample
from math import sqrt, tan, radians

MODEL_DIR = "models/"
OUTPUT_DIR = "dataset_s3/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images/")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")
IMG_SIZE = 512
MAX_PARTS_PER_SCENE = 6
SPACING = 0.1  # Space between parts


def clear_output_dirs():
    for directory in [IMG_PATH]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)


def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    return mesh


def place_parts(meshes):
    placed_meshes = []
    cursor = np.array([0.0, 0.0, 0.0])
    row_height = 0.0
    max_row_width = 2.5  # Max width before wrapping to a new row
    row_origin = np.array([0.0, 0.0, 0.0])

    for mesh in meshes:
        bbox = mesh.bounding_box.extents
        width, height, depth = bbox

        if cursor[0] + width > max_row_width:
            cursor[0] = 0.0
            cursor[2] += row_height + SPACING
            row_height = 0.0

        translation = cursor + np.array([width / 2, 0, depth / 2])
        mesh_copy = mesh.copy()
        mesh_copy.apply_translation(translation)
        placed_meshes.append(mesh_copy)

        cursor[0] += width + SPACING
        row_height = max(row_height, depth)

    return placed_meshes


def get_combined_bounds(meshes):
    all_bounds = [mesh.bounds for mesh in meshes]
    mins = np.min([b[0] for b in all_bounds], axis=0)
    maxs = np.max([b[1] for b in all_bounds], axis=0)
    return mins, maxs


def compute_camera_pose_and_distance(bounds_min, bounds_max):
    center = (bounds_min + bounds_max) / 2
    size = bounds_max - bounds_min
    diagonal = np.linalg.norm(size)

    # Isometric view (viewed from corner of bounding box)
    cam_position = center + np.array([1, 1, 1]) * diagonal
    cam_position = cam_position / np.linalg.norm(cam_position) * diagonal * 5

    # Create a simple look-at matrix
    forward = center - cam_position
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_position
    return pose


def render_scene(meshes, renderer):
    scene = pyrender.Scene(
        bg_color=[0.95, 0.95, 0.95, 1.0], ambient_light=[0.3, 0.3, 0.3, 1.0]
    )

    # Add meshes
    for mesh in meshes:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.4, 0.4, 0.4, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.7,
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(render_mesh)

    bounds_min, bounds_max = get_combined_bounds(meshes)
    cam_pose = compute_camera_pose_and_distance(bounds_min, bounds_max)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    color, _ = renderer.render(scene)
    return color


def main():
    start = timeit.default_timer()
    clear_output_dirs()
    step_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".step")]

    r = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    with open(METADATA_PATH, "w") as fj:
        assembly_id = 0
        while step_files:
            part_batch = sample(step_files, min(MAX_PARTS_PER_SCENE, len(step_files)))
            for p in part_batch:
                step_files.remove(p)

            meshes = []
            for step_file in part_batch:
                mesh = load_step_part(os.path.join(MODEL_DIR, step_file))
                meshes.append(mesh)

            placed_meshes = place_parts(meshes)
            color_img = render_scene(placed_meshes, r)

            img_name = f"assembly_{assembly_id}.png"
            img_path = os.path.join(IMG_PATH, img_name)
            Image.fromarray(color_img).save(img_path)

            json_line = {
                "image": img_name,
                "parts": [os.path.splitext(p)[0] for p in part_batch],
            }
            fj.write(json.dumps(json_line) + "\n")

            print(f"Rendered {img_name}")
            assembly_id += 1

    r.delete()
    stop = timeit.default_timer()
    print(f"Finished in {(stop - start):.2f}s. Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
