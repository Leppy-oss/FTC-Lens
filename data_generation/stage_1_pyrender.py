import os
import json
import shutil
import timeit
import trimesh
import pyrender
import numpy as np
from PIL import Image
from math import pi
from itertools import product

MODEL_DIR = "models/"
OUTPUT_DIR = "dataset_s1/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images/")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")
IMG_SIZE = 512
OBJ_RADIUS = 0.75


def clear_output_dirs():
    for directory in [IMG_PATH]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)


def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    scale = OBJ_RADIUS / np.max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh


def random_grayscale_color(min_val=0.2, max_val=0.8, tint_strength=0.0):
    base = np.random.uniform(min_val, max_val)
    r = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    g = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    b = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    return np.array([r, g, b])


def random_object_poses(n_rot=3, max_offset=0.05):
    angles = np.linspace(-45, 45, n_rot) + np.random.uniform(-5, 5, size=n_rot)
    rotations = list(product(angles, repeat=3))
    poses = []
    for yaw, pitch, roll in rotations:
        y, p, r = np.radians([yaw, pitch, roll])

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
        )
        Ry = np.array(
            [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
        )
        Rz = np.array(
            [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
        )
        R = Rz @ Ry @ Rx

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = np.random.uniform(
            -max_offset, max_offset, size=3
        )
        poses.append(pose)
    return poses


def render_single_view(mesh, object_pose, renderer):
    scene = pyrender.Scene(
        bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0), ambient_light=[0.3, 0.3, 0.3, 1.0]
    )

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.append(random_grayscale_color(0.35, 0.45), 1.0),
        metallicFactor=0.8,
        roughnessFactor=0.8,
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    node = scene.add(render_mesh, pose=object_pose)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, 0, 2]
    scene.add(cam, pose=cam_pose)

    light = pyrender.PointLight(color=np.ones(3), intensity=5)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 2, 0]
    scene.add(light, pose=light_pose)

    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5)
    scene.add(fill_light, pose=cam_pose)

    color, _ = renderer.render(scene)
    scene.remove_node(node)
    return color


def main():
    start = timeit.default_timer()
    clear_output_dirs()
    step_files = [
        f for f in os.listdir(MODEL_DIR) if f.endswith(".STEP") or f.endswith(".step")
    ]

    r = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    with open(METADATA_PATH, "w") as fj:
        for step_file in step_files:
            mesh = load_step_part(os.path.join(MODEL_DIR, step_file))
            part_name = os.path.splitext(step_file)[0]
            print(f"Rendering {part_name}")

            obj_poses = random_object_poses()

            for i, pose in enumerate(obj_poses):
                img = render_single_view(mesh, pose, r)
                img_path = os.path.join(IMG_PATH, f"{part_name}_{i}.png")
                Image.fromarray(img).save(img_path)

                json_line = {"image": f"{part_name}_{i}.png", "label": part_name}
                fj.write(json.dumps(json_line) + "\n")

    r.delete()
    stop = timeit.default_timer()
    print(f"Finished in {(stop - start):.2f}s. Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
