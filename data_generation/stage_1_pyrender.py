import os
import json
import shutil
import timeit
import trimesh
import pyrender
import numpy as np
from PIL import Image
from itertools import product

MODEL_DIR = "models_sm/"
OUTPUT_DIR = "dataset_s1_sm/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")
IMG_SIZE = 280
CAMERA_RADIUS = 4

all_xyzms = [
    v for v in product((-1, 0, 1), repeat=3)
    if v != (0, 0, 0) and sum(x != 0 for x in v) in (1, 3)
]

def clear_output_dirs():
    for directory in [IMG_PATH]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

def random_grayscale_color(min_val=0.2, max_val=0.8, tint_strength=0.0):
    base = np.random.uniform(min_val, max_val)
    r = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    g = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    b = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    return np.array([r, g, b])


def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    scale = 0.8 / np.max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh


def look_at(v_e, v_t=np.array((0, 0, 0)), up=np.array((0, 1, 0))):
    forward = v_t - v_e
    forward /= np.linalg.norm(forward)
    up = np.array([1, 0, 0]) if abs(np.dot(forward, up)) > 0.999 else up

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = up
    rot[:3, 2] = -forward

    trans = np.eye(4)
    trans[:3, 3] = v_e

    pose = trans @ rot
    return pose


def path_to_mesh(path):
    lines = []
    for entity in path.entities:
        segment = entity.discrete(path.vertices)
        for i in range(len(segment) - 1):
            lines.append(segment[i])
            lines.append(segment[i + 1])

    lines_np = np.array(lines, dtype=np.float32)

    line_mesh = pyrender.Primitive(
        positions=lines_np,
        mode=1,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0, 0, 0),
            metallicFactor=0.8,
            roughnessFactor=0.8,
        ),
    )
    return pyrender.Mesh([line_mesh])


def generate_camera_positions(r=CAMERA_RADIUS, n_elev=16, n_azi=16):
    positions = []

    for i_a in range(n_azi):
        az = i_a * 2 * np.pi / n_azi
        for i_e in range(n_elev):
            el = i_e * np.pi / n_elev - np.pi / 2
            x = r * np.cos(el) * np.cos(az)
            y = r * np.cos(el) * np.sin(az)
            z = r * np.sin(el)
            positions.append(np.array((x, y, z)))

    return positions


def generate_camera_poses(v_t_center=np.array([0, 0, 0]), offset_scale=0.3):
    poses = []
    positions = generate_camera_positions()

    for v_e in positions:
        v_t = v_t_center + np.random.uniform(-offset_scale, offset_scale, 3)
        forward = v_t - v_e
        forward /= np.linalg.norm(forward)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        true_up = np.cross(right, forward)

        rot = np.eye(4)
        rot[:3, 0] = right
        rot[:3, 1] = true_up
        rot[:3, 2] = -forward  # openGL std

        trans = np.eye(4)
        trans[:3, 3] = v_e

        pose = trans @ rot
        poses.append(pose)

    return poses


def render(mesh, camera_pose, renderer):
    scene = pyrender.Scene(bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0), ambient_light=[0.3, 0.3, 0.3, 1.0])

    m = mesh.copy()
    m.merge_vertices()
    scene.add(
        pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.append(random_grayscale_color(0.35, 0.45), 1.0),
            metallicFactor=0.8,
            roughnessFactor=0.8,
        ), smooth=False)
    )

    scene.add(path_to_mesh(trimesh.load_path(m.vertices[m.face_adjacency_edges[m.face_adjacency_angles >= np.radians(90)]])))

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(cam, pose=camera_pose)

    for xyzm in all_xyzms:
        light_pose = look_at(CAMERA_RADIUS * np.array(xyzm, dtype=np.dtypes.Float64DType))
        light = pyrender.DirectionalLight(intensity=2.5)
        scene.add(light, pose=light_pose)

    color, _ = renderer.render(scene)
    return color


def main():
    start = timeit.default_timer()
    clear_output_dirs()
    steps = [f for f in os.listdir(MODEL_DIR) if f.endswith(".STEP") or f.endswith(".step")]

    r = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    with open(METADATA_PATH, "w") as fj:
        for f in steps:
            mesh = load_step_part(os.path.join(MODEL_DIR, f))
            fname = os.path.splitext(f)[0]
            print(f"Rendering {fname}")

            camera_poses = generate_camera_poses()

            for i, pose in enumerate(camera_poses):
                img = render(mesh, pose, r)
                img_path = os.path.join(IMG_PATH, f"{fname}_{i}.png")
                Image.fromarray(img).save(img_path)

                json_line = {"file_name": f"images/{fname}_{i}.png", "label": fname}
                fj.write(json.dumps(json_line) + "\n")

    r.delete()
    stop = timeit.default_timer()

    print(f"Finished all stage 1 pyrenders in {(stop - start):.2f}s. Saved metadata to {METADATA_PATH}.")


if __name__ == "__main__":
    main()
