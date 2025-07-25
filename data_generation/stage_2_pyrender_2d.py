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
METADATA_FILE = "metadata.jsonl"

SPLIT = {
    "train": 0.8,
    "test": 0.2
}

IMG_SIZE = 280
SMALL_PARTS_PER_SCENE = (3, 4, 1, 10)  # mu, sigma, min, max
SCENE_COUNT = 100000

all_xyzms = [
    v for v in product((-1, 0, 1), repeat=3)
    if v != (0, 0, 0) and sum(x != 0 for x in v) in (1, 3)
]


def clear_output_dirs():
    for path in [os.path.join(OUTPUT_DIR, p) for p in SPLIT.keys()]:
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


GRID_SIZE = 24

def rotate_mesh(mesh, by=None):
    rot_x = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    y_angle = by or random.choice([0, 90, 180, 270])
    rot_y = trimesh.transformations.rotation_matrix(np.radians(y_angle), [0, 1, 0])
    transform = trimesh.transformations.concatenate_matrices(rot_y, rot_x)
    mesh = mesh.copy()
    mesh.apply_transform(transform)
    return mesh, transform, y_angle

def get_footprint(mesh):
    bbox = mesh.bounding_box.extents
    x_cells = int(round(bbox[0] / GRID_SIZE))
    z_cells = int(round(bbox[2] / GRID_SIZE))
    return (x_cells, z_cells)

def cornerify(mesh, outline):
    bbox = mesh.bounding_box.bounds
    min_corner = bbox[0]
    mesh.apply_translation(-min_corner)
    outline.apply_translation(-min_corner)
    
def get_min_corner(mesh):
    return mesh.bounding_box.bounds[1]
    
def get_adjacent_positions(occupied, footprint): # just a lil graph theory
    adj_positions = set()
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for (x,y) in occupied:
        for dx, dz in directions:
            cx = x + dx
            cz = y + dz

            candidate_cells = set(
                (cx + i, cz + j)
                for i in range(footprint[0])
                for j in range(footprint[1])
            )

            if not candidate_cells & occupied:
                adj_positions.add((cx, cz))

    return list(adj_positions)

def assemble(meshes, outlines=None):
    if outlines is None:
        outlines = [None] * len(meshes)

    placed_meshes = []
    placed_outlines = []
    transforms = []
    y_rotations = []
    grid_positions = []

    # first anchor at origin
    mesh, tf, y_angle = rotate_mesh(meshes[0].copy())
    outline = outlines[0].copy() if outlines[0] is not None else None
    if outline:
        outline.apply_transform(tf)
    footprint = get_footprint(mesh)

    cornerify(mesh, outline)
    placed_meshes.append(mesh)
    placed_outlines.append(outline)
    transforms.append(tf)
    y_rotations.append(y_angle)
    grid_positions.append((0,0))  # first mesh at origin

    occupied = set((x, z) for x in range(footprint[0]) for z in range(footprint[1]))

    for mesh, outline in zip(meshes[1:], outlines[1:]):
        success = False
        for _ in range(100):
            mesh_rot, tf, y_angle = rotate_mesh(mesh)
            footprint = get_footprint(mesh_rot)

            candidates = get_adjacent_positions(occupied, footprint)
            if not candidates:
                continue

            pos = random.choice(candidates)
            x_grid, z_grid = pos

            mesh_copy = mesh_rot.copy()
            outline_copy = outline.copy() if outline is not None else None
            if outline_copy:
                outline_copy.apply_transform(tf)
            cornerify(mesh_copy, outline_copy)
            mesh_copy.apply_translation([x_grid * GRID_SIZE, 0, z_grid * GRID_SIZE])

            if outline_copy:
                outline_copy.apply_translation([x_grid * GRID_SIZE, 0, z_grid * GRID_SIZE])

            new_cells = set(
                (x_grid + dx, z_grid + dz)
                for dx in range(footprint[0])
                for dz in range(footprint[1])
            )
            if new_cells & occupied:
                continue # overlap, try again

            occupied |= new_cells
            placed_meshes.append(mesh_copy)
            placed_outlines.append(outline_copy)
            transforms.append(tf)
            y_rotations.append(y_angle)
            grid_positions.append((x_grid, z_grid))
            success = True
            break

        if not success:
            raise RuntimeError(f"Could not place mesh after 100 attempts")

    # recentering
    centroids = np.array([m.centroid + get_min_corner(m) for m in placed_meshes])
    assembly_centroid = np.mean(centroids, axis=0)
    distances = np.linalg.norm(centroids - assembly_centroid, axis=1)
    anchor_idx = int(np.argmin(distances))
    offset = -placed_meshes[anchor_idx].centroid

    for mesh, outline in zip(placed_meshes, placed_outlines):
        mesh.apply_translation(offset)
        outline.apply_translation(offset)

    final_transforms = []
    for i, (x_grid, z_grid) in enumerate(grid_positions):
        min_corner = get_min_corner(rotate_mesh(meshes[i], y_rotations[i])[0])
        final_x = x_grid * GRID_SIZE + offset[0] + min_corner[0]
        final_z = z_grid * GRID_SIZE + offset[2] + min_corner[2]

        final_transforms.append({
            "translation": {"x": float(f"{final_x:z.2f}"), "y": float(f"{final_z:z.2f}")},
            "rotation": y_rotations[i]
        })

    return placed_meshes, placed_outlines, final_transforms


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
    size = np.linalg.norm(bounds[1] - bounds[0])

    direction = xyzm
    direction = direction / np.linalg.norm(direction)

    cam_dist = size * 0.7  # adjust zoom level
    cam_position = direction * cam_dist
    cam_pose = look_at(cam_position)

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

    camera = pyrender.OrthographicCamera(cam_dist, cam_dist, zfar=1000)

    scene.add(camera, pose=cam_pose)

    color, _ = r.render(scene)
    return color

    
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

    return converted_data["transforms"]


process = psutil.Process(os.getpid())
clear_output_dirs()

steps = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".step")]
meshes_cache = {}
outlines_cache = {}
r = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
peak_memory_usage = -1
t_mesh_load_time = 0
t_outline_load_time = 0
t_render_time = 0
last_memory_usage = process.memory_info().rss / (1024 * 1024)

warnings.filterwarnings("ignore", category=UserWarning)  # ignore annoying scipy gimbal lock warnings

def do_split(split_name):
    global steps, meshes_cache, outlines_cache, r, peak_memory_usage, t_mesh_load_time, t_outline_load_time, t_render_time, last_memory_usage
    with open(os.path.join(OUTPUT_DIR, split_name, METADATA_FILE), "w") as meta_file:
        for i in range(int(SCENE_COUNT * SPLIT[split_name])):
            file_names = np.random.choice(
                steps,
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

            part_names = [os.path.splitext(f)[0] for f in file_names]
            print(f"[{i+1}/{int(SCENE_COUNT * SPLIT[split_name])} {split_name}] Parts (n={len(file_names)}): {', '.join(part_names)}")

            mesh_load_start = timeit.default_timer()
            meshes = [
                meshes_cache.get(f, False) or meshes_cache.setdefault(f, load_step_part(os.path.join(MODEL_DIR, f)))
                for f in file_names
            ]
            mesh_load_time = timeit.default_timer() - mesh_load_start
            t_mesh_load_time += mesh_load_time

            outline_load_start = timeit.default_timer()

            def load_outline(f, mesh):
                def compute_outline(mesh):
                    m = mesh.copy()
                    m.merge_vertices()
                    return trimesh.load_path(m.vertices[m.face_adjacency_edges[m.face_adjacency_angles >= np.radians(90)]])
                
                return outlines_cache.get(f, False) or outlines_cache.setdefault(f, compute_outline(mesh))
            
            outlines = [load_outline(f, mesh) for f, mesh in zip(file_names, meshes)]
            outline_load_time = timeit.default_timer() - outline_load_start
            t_outline_load_time += outline_load_time

            render_start = timeit.default_timer()
            meshes_transformed, outlines_transformed, indexed_transforms = assemble(meshes, outlines)

            img = render(meshes_transformed, outlines_transformed, r, [0, 1, 0])
            img_path = f"{i:05d}.png"
            Image.fromarray(img).save(os.path.join(OUTPUT_DIR, split_name, img_path))
            
            named_transforms = [
                {
                    "name": os.path.splitext(n)[0],
                    "translation": t["translation"],
                    "rotation": t["rotation"],
                } for t, n in zip(indexed_transforms, file_names)
            ]
            
            metadata = {
                "image": os.path.join("images", img_path),
                "label": json.dumps(convert_and_sort({"transforms": named_transforms}))
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

        meta_file.close()

do_split("train")
do_split("test")

r.delete()
print(f"Finished in {(t_mesh_load_time + t_outline_load_time + t_render_time):.2f}s ({t_mesh_load_time:.2f} mesh / {t_outline_load_time:.2f} outline / {t_render_time:.2f} render). Peak memory consumption {peak_memory_usage:.2f} MiB.")