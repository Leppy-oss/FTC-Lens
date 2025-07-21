import os
import json
import shutil
import timeit
import random
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


def random_grayscale_color(min_val=0.2, max_val=0.8, tint_strength=0.0):
    base = np.random.uniform(min_val, max_val)
    r = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    g = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    b = np.clip(base + np.random.uniform(-tint_strength, tint_strength), 0, 1)
    return np.array([r, g, b])


def look_at(v_e, v_t=np.array((0, 0, 0)), up=np.array((0, 1, 0))):
    forward = v_t - v_e
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = true_up
    rot[:3, 2] = -forward

    trans = np.eye(4)
    trans[:3, 3] = v_e

    pose = trans @ rot
    return pose


def intersects(bounds1, bounds2):
    return np.all(bounds1[0] < bounds2[1]) and np.all(bounds1[1] > bounds2[0])


def get_random_face_transform(target_mesh, source_mesh):
    target_box = target_mesh.bounding_box_oriented
    source_box = source_mesh.bounding_box_oriented

    target_face = random.randint(0, 5)
    source_face = random.randint(0, 5)

    target_min, target_max = target_box.bounds
    source_min, source_max = source_box.bounds

    def get_face_vector(idx, min_, max_):
        axis = idx % 3
        direction = -1 if idx < 3 else 1
        face_center = np.array(
            [
                (
                    min_[i]
                    if direction == -1 and i == axis
                    else (
                        max_[i]
                        if direction == 1 and i == axis
                        else (min_[i] + max_[i]) / 2
                    )
                )
                for i in range(3)
            ]
        )
        normal = np.eye(3)[axis] * direction
        return face_center, normal

    t_face_center, t_normal = get_face_vector(target_face, target_min, target_max)
    s_face_center, s_normal = get_face_vector(source_face, source_min, source_max)

    v1 = s_normal / np.linalg.norm(s_normal)
    v2 = -t_normal / np.linalg.norm(t_normal)

    axis = np.cross(v1, v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    if np.linalg.norm(axis) < 1e-6:
        R_mat = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        R_mat = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]

    s_transformed_face = R_mat @ s_face_center
    translation = t_face_center - s_transformed_face

    rotation_euler = R.from_matrix(R_mat).as_euler("xyz", degrees=True)

    return translation, rotation_euler


def assemble_meshes(mesh_list, max_attempts=100):
    placed = []
    poses = []  # (position, rotation_euler_deg)

    mesh0 = mesh_list[0].copy()
    placed.append(mesh0)
    meshes_transformed = [mesh0]
    poses.append((np.zeros(3), np.zeros(3)))  # origin at (0,0,0) with no rotation
    origin_index = 0

    for mesh in mesh_list[1:]:
        added = False
        attempts = 0

        while not added and attempts < max_attempts:
            target = random.choice(placed)
            position, rotation_euler = get_random_face_transform(target, mesh)

            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R.from_euler("xyz", rotation_euler, degrees=True).as_matrix()
            T[:3, 3] = position

            mesh_transformed = mesh.copy()
            mesh_transformed.apply_transform(T)

            if not any(intersects(mesh_transformed.bounds, p.bounds) for p in placed):
                placed.append(mesh_transformed)
                meshes_transformed.append(mesh_transformed)
                poses.append((position, rotation_euler))
                added = True
            attempts += 1

    print(f"Using {len(meshes_transformed)} total meshes")

    return meshes_transformed, poses, origin_index


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


def render_scene_pyrender(meshes_transformed: list[trimesh.Trimesh], r: pyrender.Renderer):
    full_mesh = trimesh.Scene(meshes_transformed).dump(concatenate=True)
    bounds = full_mesh.bounds
    center = full_mesh.centroid
    size = np.linalg.norm(bounds[1] - bounds[0])

    direction = np.array([1, 1, 1])
    direction = direction / np.linalg.norm(direction)

    camera_distance = size * 1.5  # adjust zoom level
    cam_position = center + direction * camera_distance
    cam_pose = look_at(cam_position, center)

    scene = pyrender.Scene(
        bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0),
        ambient_light=[0.3, 0.3, 0.3, 1.0],
    )
    for mesh in meshes_transformed:
        m = mesh.copy()
        m.merge_vertices()
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.append(random_grayscale_color(0.35, 0.45), 1.0),
            metallicFactor=0.8,
            roughnessFactor=0.8,
        )
        pymesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(pymesh)
        scene.add(
            path_to_mesh(
                trimesh.load_path(
                    m.vertices[
                        m.face_adjacency_edges[
                            m.face_adjacency_angles
                            >= np.radians(
                                90 - max(0, min((200000 - len(m.vertices)) * 0.001, 80))
                            )
                        ]
                    ]
                )
            )
        )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # camera = pyrender.OrthographicCamera(xmag=size, ymag=size)  # Optional for isometric flatness

    scene.add(camera, pose=cam_pose)

    # light at the camera position
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    color, _ = r.render(scene)
    return color

def export_transforms(poses, origin_index, mesh_names):
    data = {
        "origin_name": mesh_names[origin_index],
        "transforms": [],
    }

    for i, (pos, rot) in enumerate(poses):
        entry = {
            "name": mesh_names[i],
            "Position": pos,
            "Rotation": rot,
        }
        data["transforms"].append(entry)

    return data


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
