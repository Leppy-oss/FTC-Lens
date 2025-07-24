import os
import gc
import json
import shutil
import psutil
import timeit
import random
import trimesh
import pyrender
import warnings
import numpy as np
from PIL import Image
from itertools import product
from large_models import large_models as large_step_files
from scipy.spatial.transform import Rotation as R

MODEL_DIR = "models_sm/"
OUTPUT_DIR = "dataset_s2_sm/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images/")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")

IMG_SIZE = 1024
SMALL_PARTS_PER_SCENE = (4, 6, 2, 20)  # mu, sigma, min, max
LARGE_PARTS_PER_SCENE = (1, 2, 0, 4)  # mu, sigma, min, max
VIEWS_PER_SCENE = (1, 2, 1, 4)  # mu, sigma, min, max
SCENE_COUNT = 30000

LARGE_PART_CHANCE = 0.0

all_xyzms = [
    v for v in product((-1, 0, 1), repeat=3)
    if v != (0, 0, 0) and sum(x != 0 for x in v) in (1, 3)
]


def clear_output_dirs():
    for path in [IMG_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def load_step_part(filepath):
    mesh = trimesh.load_mesh(filepath)
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


def assemble(mesh_list, outline_list, max_attempts=100):
    placed = []
    poses = []

    rot_euler_first = np.random.choice([0, 90, 180, 270], size=3)
    T0 = np.eye(4)
    T0[:3, :3] = R.from_euler("xyz", rot_euler_first, degrees=True).as_matrix()

    mesh0 = mesh_list[0].copy()
    mesh0.apply_transform(T0)
    meshes_transformed = [mesh0]

    outline0 = outline_list[0].copy()
    outline0.apply_transform(T0)
    outlines_transformed = [outline0]

    placed.append(mesh0)
    poses.append((np.zeros(3), rot_euler_first))

    for mesh, outline in zip(mesh_list[1:], outline_list[1:]):
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

                outline_transformed = outline.copy()
                outline_transformed.apply_transform(T)
                outlines_transformed.append(outline_transformed)

                poses.append((position, rotation_euler))
                added = True

            attempts += 1

    all_bounds = np.vstack([m.bounds for m in meshes_transformed])
    com = (np.min(all_bounds, axis=0) + np.max(all_bounds, axis=0)) / 2 # centroid

    mesh_centers = np.array([m.bounding_box.centroid for m in meshes_transformed])
    distances = np.linalg.norm(mesh_centers - com, axis=1)
    origin_idx = np.argmin(distances)
    origin = mesh_centers[origin_idx]

    offset = origin

    for i, (m, o) in enumerate(zip(meshes_transformed, outlines_transformed)):
        m.apply_translation(-offset)
        o.apply_translation(-offset)
        pos, rot = poses[i]
        new_pos = pos - offset
        poses[i] = (new_pos, rot)

    return meshes_transformed, outlines_transformed, poses, origin_idx


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


def render(m_transformed: list[trimesh.Trimesh], o_transformed, r: pyrender.Renderer, xyzm: tuple[int]):
    full_mesh = trimesh.Scene(m_transformed).to_geometry()
    bounds = full_mesh.bounds
    center = full_mesh.centroid
    size = np.linalg.norm(bounds[1] - bounds[0])

    direction = xyzm
    direction = direction / np.linalg.norm(direction)

    cam_dist = size * 1.25  # adjust zoom level
    cam_position = center + direction * cam_dist
    cam_pose = look_at(cam_position, center)

    scene = pyrender.Scene(bg_color=np.append(random_grayscale_color(0.9, 0.975), 1.0), ambient_light=[0.3, 0.3, 0.3, 1.0])

    for xyzm in all_xyzms:
        light_pose = look_at(cam_dist * np.array(xyzm, dtype=np.dtypes.Float64DType))
        light = pyrender.DirectionalLight(intensity=2.5)
        scene.add(light, pose=light_pose)

    for mesh, outline in zip(m_transformed, o_transformed):
        scene.add(
            pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.append(random_grayscale_color(0.35, 0.45), 1.0),
                metallicFactor=0.8,
                roughnessFactor=0.8,
            ), smooth=False)
        )
        scene.add(path_to_mesh(outline))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

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
            return [round(v, 2) for v in obj.tolist()]
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

process = psutil.Process(os.getpid())
clear_output_dirs()

all_step_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".step")]
small_step_files = np.setdiff1d(all_step_files, large_step_files)
meshes_cache = {}
outlines_cache = {}
r = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
peak_memory_usage = -1
t_mesh_load_time = 0
t_outline_load_time = 0
t_render_time = 0
last_memory_usage = process.memory_info().rss / (1024 * 1024)

warnings.filterwarnings("ignore", category=UserWarning)  # ignore annoying scipy gimbal lock warnings
with open(METADATA_PATH, "w") as meta_file:
    for i in range(SCENE_COUNT):
        file_names = np.random.choice(
            small_step_files,
            np.clip(
                round(
                    random.normalvariate(
                        SMALL_PARTS_PER_SCENE[0], SMALL_PARTS_PER_SCENE[1]
                    )
                ),
                SMALL_PARTS_PER_SCENE[2],
                SMALL_PARTS_PER_SCENE[3],
            ),
            replace=True,
        )

        large_file_names = []
        if random.random() < LARGE_PART_CHANCE:
            large_file_names = np.random.choice(
                large_step_files,
                np.clip(
                    round(
                        random.normalvariate(
                            LARGE_PARTS_PER_SCENE[0], LARGE_PARTS_PER_SCENE[1]
                        )
                    ),
                    LARGE_PARTS_PER_SCENE[2],
                    LARGE_PARTS_PER_SCENE[3],
                ),
                replace=True,
            )

        file_names = np.concatenate((file_names, large_file_names))
        part_names = [os.path.splitext(f)[0] for f in file_names]
        print(f"[{i+1}/{SCENE_COUNT}] Parts (n={len(file_names) - len(large_file_names)}S+{len(large_file_names)}L): {', '.join(part_names)}")

        mesh_load_start = timeit.default_timer()
        meshes = [
            load_step_part(os.path.join(MODEL_DIR, f)) if f in large_file_names 
            else meshes_cache.get(f, False) or meshes_cache.setdefault(f, load_step_part(os.path.join(MODEL_DIR, f)))
            for f in file_names
        ]
        mesh_load_time = timeit.default_timer() - mesh_load_start
        t_mesh_load_time += mesh_load_time

        outline_load_start = timeit.default_timer()

        def load_outline(f, mesh):
            def compute_outline(mesh):
                m = mesh.copy()
                m.merge_vertices()
                thresh = 10 if len(m.vertices) < 100000 else 30 if len(m.vertices) < 150000 else 90
                return trimesh.load_path(m.vertices[m.face_adjacency_edges[m.face_adjacency_angles >= np.radians(thresh)]])
            
            return compute_outline(mesh) if f in large_file_names else outlines_cache.get(f, False) or outlines_cache.setdefault(f, compute_outline(mesh))
        
        outlines = [load_outline(f, mesh) for f, mesh in zip(file_names, meshes)]
        outline_load_time = timeit.default_timer() - outline_load_start
        t_outline_load_time += outline_load_time

        render_start = timeit.default_timer()
        meshes_transformed, outlines_transformed, poses, origin_index = assemble(meshes, outlines)

        xyzms = [(1, 1, 1)] + random.sample(
            [xyzm for xyzm in all_xyzms if xyzm != (1, 1, 1)],
            np.clip(
                round(random.normalvariate(VIEWS_PER_SCENE[0], VIEWS_PER_SCENE[1])),
                VIEWS_PER_SCENE[2],
                VIEWS_PER_SCENE[3],
            ),
        )
        data = export_transforms(poses, origin_index, part_names)

        img_paths = []

        for ixyzm, xyzm in enumerate(xyzms):
            img = render(meshes_transformed, outlines_transformed, r, xyzm)
            img_path = f"scene_{i:05d}_view_{ixyzm}.png"
            img_paths.append(img_path)
            Image.fromarray(img).save(os.path.join(IMG_PATH, img_path))

        metadata = {
            "file_names": img_paths,
            "origin": data["origin_name"],
            "label": convert_and_sort(data["transforms"]),
        }

        meta_file.write(json.dumps(metadata) + "\n")
        meta_file.flush() # write in real time
        render_time = timeit.default_timer() - render_start
        t_render_time += render_time

        memory_usage = process.memory_info().rss / (1024 * 1024)
        print(f"Rendered all scenes in {(mesh_load_time + outline_load_time + render_time):.2f}s ({mesh_load_time:.2f} mesh / {outline_load_time:.2f} outline / {render_time:.2f} render). Current memory consumption {memory_usage:.2f} MiB ({((memory_usage - last_memory_usage) * 1024):.2f} KiB delta). Current cache size is {len(meshes_cache)}.")
        last_memory_usage = memory_usage
        peak_memory_usage = max(peak_memory_usage, memory_usage)

        del meshes_transformed, outlines_transformed
        gc.collect()

r.delete()
print(f"Finished in {(t_mesh_load_time + t_outline_load_time + t_render_time):.2f}s ({t_mesh_load_time:.2f} mesh / {t_outline_load_time:.2f} outline / {t_render_time:.2f} render). Peak memory consumption {peak_memory_usage:.2f} MiB.")