import os
import json
import shutil
import timeit
import random
import trimesh
import pyrender
import numpy as np
from PIL import Image

MODEL_DIR = "models/"
OUTPUT_DIR = "dataset_s2/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images/")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")

IMG_SIZE = 512
PARTS_PER_SCENE = (2, 4)
SCENE_COUNT = 100
MAX_Z_SPREAD = 1.0  # Units of height Z spread limit

CAPTION_TEMPLATES = [
    "The assembly includes {}.",
    "Present in this subsystem are {}.",
    "{} are shown in this view.",
    "You can see {} assembled together.",
    "{} make up this component group."
]


def clear_output_dirs():
    for path in [IMG_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def random_grayscale_color(min_val=0.2, max_val=0.8, tint_strength=0.0):
    base = np.random.uniform(min_val, max_val)
    r = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    g = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    b = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    return np.array([r, g, b])


def random_limited_rotation(max_axes=2):
    axes = ['x', 'y', 'z']
    chosen_axes = random.sample(axes, k=random.randint(1, max_axes))
    rot = np.eye(4)
    for ax in chosen_axes:
        angle = np.radians(random.choice(range(0, 360, 15)))
        if ax == 'x':
            rot = rot @ trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        elif ax == 'y':
            rot = rot @ trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        elif ax == 'z':
            rot = rot @ trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    return rot


def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    return mesh


def tightly_place_parts(meshes, z_clamp_range=MAX_Z_SPREAD, spacing_factor=1.2):
    placed = []
    transforms = []

    radius_step = max(np.max([m.extents.max() for m in meshes]), 0.1) * spacing_factor
    angle_step = 2 * np.pi / max(len(meshes), 1)

    for i, mesh in enumerate(meshes):
        angle = i * angle_step
        radius = radius_step * (i // 3 + 1)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(-z_clamp_range / 2, z_clamp_range / 2)

        # Analyze extents
        extents = mesh.extents
        axes = ["x", "y", "z"]
        thin_axis = np.argmin(extents)

        # Make sure thinnest axis doesn't point toward camera (+Z)
        correction = np.eye(4)
        if thin_axis == 2:  # If Z is thinnest, rotate to make it horizontal
            correction = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        elif thin_axis == 1:
            correction = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        # else: already facing reasonably

        # Additional random rotation for variety (excluding axis that aligns with camera)
        rand_rot = random_limited_rotation(max_axes=1)

        transform = trimesh.transformations.translation_matrix([x, y, z])
        transform = transform @ correction @ rand_rot

        transformed = mesh.copy()
        transformed.apply_transform(transform)

        min_z = transformed.bounds[0][2]
        if min_z > 0:
            z_shift = trimesh.transformations.translation_matrix([0, 0, -min_z])
            transform = z_shift @ transform
            transformed.apply_transform(z_shift)

        placed.append(transformed)
        transforms.append(transform)

    return placed, transforms


def compute_camera_distance(meshes, fov=np.pi / 4.0, margin=1.05):
    scene = trimesh.Scene(meshes)
    bounds = scene.bounds
    size = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2.0
    radius = np.linalg.norm(size) / 2.0
    distance = radius / np.tan(fov / 2.0)
    return distance * margin, center


def render_scene(meshes, renderer):
    scene = pyrender.Scene(
        bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0),
        ambient_light=[0.3, 0.3, 0.3, 1.0]
    )

    for mesh in meshes:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.append(random_grayscale_color(0.35, 0.45), 1.0),
            metallicFactor=0.8,
            roughnessFactor=0.8,
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(render_mesh)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_distance, center = compute_camera_distance(meshes)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [center[0], center[1], cam_distance]
    scene.add(cam, pose=cam_pose)

    light = pyrender.PointLight(color=np.ones(3), intensity=5)
    scene.add(light, pose=cam_pose)

    return renderer.render(scene)[0]


def generate_caption(part_names):
    if len(part_names) == 1:
        body = part_names[0]
    else:
        body = ", ".join(part_names[:-1]) + f", and {part_names[-1]}"
    return random.choice(CAPTION_TEMPLATES).format(body)


def main():
    start = timeit.default_timer()
    clear_output_dirs()

    all_step_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".step")]
    meshes_cache = {}
    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)

    with open(METADATA_PATH, "w") as meta_file:
        for i in range(SCENE_COUNT):
            part_files = random.sample(all_step_files, random.randint(*PARTS_PER_SCENE))
            part_names = [os.path.splitext(f)[0] for f in part_files]

            print(f"[{i+1}/{SCENE_COUNT}] Parts: {', '.join(part_names)}")

            meshes = []
            for file in part_files:
                if file not in meshes_cache:
                    meshes_cache[file] = load_step_part(os.path.join(MODEL_DIR, file))
                meshes.append(meshes_cache[file])

            placed_meshes, _ = tightly_place_parts(meshes)

            img = render_scene(placed_meshes, renderer)
            img_path = os.path.join(IMG_PATH, f"scene_{i:04d}.png")
            Image.fromarray(img).save(img_path)

            metadata = {
                "image": os.path.basename(img_path),
                "label": generate_caption(part_names),
                "parts": part_names
            }
            meta_file.write(json.dumps(metadata) + "\n")

    renderer.delete()
    elapsed = timeit.default_timer() - start
    print(f"\n✅ Done in {elapsed:.2f}s. Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
