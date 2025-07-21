import os
import json
import shutil
import timeit
import random
import trimesh
import pyrender
import numpy as np
from PIL import Image
from itertools import product
from scipy.spatial.transform import Rotation as R

MODEL_DIR = "models/"
OUTPUT_DIR = "dataset_s2/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images/")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")

IMG_SIZE = 512
PARTS_PER_SCENE = (3, 15)
VIEWS_PER_SCENE = (1, 4)
SCENE_COUNT = 100
MAX_Z_SPREAD = 1.0  # Units of height Z spread limit

xyzms = xyzms = [v for v in product((-1, 0, 1), repeat=3) if v != (0, 0, 0) and sum(x != 0 for x in v) in (1, 3)]


def clear_output_dirs():
    for path in [IMG_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1000)
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


def intersects(bounds1, bounds2):
    return np.all(bounds1[0] < bounds2[1]) and np.all(bounds1[1] > bounds2[0])


def rand_face_transform(target_mesh, source_mesh):
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


def assemble(mesh_list, max_attempts=100):
    placed = []
    poses = []

    rot_euler_first = np.random.choice([0, 90, 180, 270], size=3)
    T0 = np.eye(4)
    T0[:3, :3] = R.from_euler("xyz", rot_euler_first, degrees=True).as_matrix()

    mesh0 = mesh_list[0].copy()
    mesh0.apply_transform(T0)
    placed.append(mesh0)
    meshes_transformed = [mesh0]
    poses.append(
        (np.zeros(3), rot_euler_first)
    )

    for mesh in mesh_list[1:]:
        added = False
        attempts = 0

        while not added and attempts < max_attempts:
            target = random.choice(placed)
            position, rotation_euler = rand_face_transform(target, mesh)

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

    all_bounds = np.vstack([m.bounds for m in meshes_transformed])
    overall_centroid = (np.min(all_bounds, axis=0) + np.max(all_bounds, axis=0)) / 2

    mesh_centers = np.array([m.bounding_box.centroid for m in meshes_transformed])
    distances = np.linalg.norm(mesh_centers - overall_centroid, axis=1)
    reference_idx = np.argmin(distances)
    reference_center = mesh_centers[reference_idx]

    offset = reference_center

    for i, m in enumerate(meshes_transformed):
        m.apply_translation(-offset)
        pos, rot = poses[i]
        new_pos = pos - offset
        poses[i] = (new_pos, rot)

    print(
        f"Using {len(meshes_transformed)} total meshes, centered on mesh index {reference_idx}"
    )

    return meshes_transformed, poses, reference_idx


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


def render_scene(meshes_transformed: list[trimesh.Trimesh], r: pyrender.Renderer, xyzm: tuple[int]):
    full_mesh = trimesh.Scene(meshes_transformed).to_geometry()
    bounds = full_mesh.bounds
    center = full_mesh.centroid
    size = np.linalg.norm(bounds[1] - bounds[0])

    direction = xyzm
    direction = direction / np.linalg.norm(direction)

    cam_dist = size * 1.25  # adjust zoom level
    cam_position = center + direction * cam_dist
    cam_pose = look_at(cam_position, center)

    scene = pyrender.Scene(
        bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0),
        ambient_light=[0.3, 0.3, 0.3, 1.0],
    )

    for xyzm in xyzms:
        light_pose = look_at(cam_dist * np.array(xyzm, dtype=np.dtypes.Float64DType))
        light = pyrender.DirectionalLight(intensity=2.5)
        scene.add(light, pose=light_pose)

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
                                10 if len(m.vertices) < 75000 else 90
                            )
                        ]
                    ]
                )
            )
        )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # camera = pyrender.OrthographicCamera(xmag=size, ymag=size)

    scene.add(camera, pose=cam_pose)

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
            "position": pos,
            "rotation": rot,
        }
        data["transforms"].append(entry)

    return data


def convert_and_sort(data):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return [round(v, 4) for v in obj.tolist()]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj

    converted_data = convert(data)

    if "transforms" in converted_data:
        converted_data["transforms"] = sorted(
            converted_data["transforms"], key=lambda x: x["name"]
        )

    return json.dumps(converted_data)


def main():
    start = timeit.default_timer()
    clear_output_dirs()

    all_step_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".step")]
    meshes_cache = {}
    r = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)

    with open(METADATA_PATH, "w") as meta_file:
        for i in range(SCENE_COUNT):
            file_names = random.sample(all_step_files, random.randint(*PARTS_PER_SCENE))
            part_names = [os.path.splitext(f)[0] for f in file_names]

            print(f"[{i+1}/{SCENE_COUNT}] Parts (n={len(part_names)}): {', '.join(part_names)}")

            meshes = [
                (
                    load_step_part(os.path.join(MODEL_DIR, f))
                    if f not in meshes_cache
                    else meshes_cache[f]
                )
                for f in file_names
            ]
            print("Loaded meshes")

            assembled_scene, poses, origin_index = assemble(meshes)
            print("Assembled scene")

            xyzms_to_render = [(1, 1, 1)] + random.sample(
                [xyzm for xyzm in xyzms if xyzm != (1, 1, 1)],
                random.randint(*VIEWS_PER_SCENE),
            )
            data = export_transforms(poses, origin_index, part_names)

            img_paths = []

            for ixyzm, xyzm in enumerate(xyzms_to_render):
                img = render_scene(assembled_scene, r, xyzm)
                img_path = f"scene_{i:05d}_view_{ixyzm}.png"
                img_paths.append(img_path)
                Image.fromarray(img).save(os.path.join(IMG_PATH, img_path))

            metadata = {
                "file_names": img_paths,
                "origin": data["origin_name"],
                "label": convert_and_sort(data["transforms"]),
            }
            meta_file.write(json.dumps(metadata) + "\n")

            break

    r.delete()
    stop = timeit.default_timer()
    print(f"Finished in {(stop - start):.2f}s. Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
