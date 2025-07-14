import os
import json
import shutil
import timeit
import trimesh
import pyrender
import numpy as np
from PIL import Image

MODEL_DIR = "models/"
OUTPUT_DIR = "dataset_s1/"
IMG_PATH = os.path.join(OUTPUT_DIR, "images")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")
IMG_SIZE = 512
CAMERA_RADIUS = 0.05

def clear_output_dirs():
    for directory in [IMG_PATH]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

def load_step_part(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    scale = 0.8 / np.max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh

def render_single_view(mesh, camera_pose, renderer):
    scene = pyrender.Scene(bg_color=[0.95, 0.95, 0.95, 1.0], ambient_light=[0.3, 0.3, 0.3, 1.0])

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.4, 0.4, 0.4, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(render_mesh)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(cam, pose=camera_pose)

    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 2, 0]
    light = pyrender.PointLight(color=np.ones(3), intensity=5)
    scene.add(light, pose=light_pose)

    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5)
    scene.add(fill_light, pose=camera_pose)

    color, _ = renderer.render(scene)
    return color


def generate_orbit_camera_poses(num_views=8, radius=1.0, center=np.array([0.0, 0.0, 0.0])):
    base_pose = np.array([
        [0.0,  -np.sqrt(2)/2,  np.sqrt(2)/2, center[0] + radius],
        [1.0,   0.0,            0.0,         center[1]],
        [0.0,   np.sqrt(2)/2,   np.sqrt(2)/2, center[2] + radius],
        [0.0,   0.0,            0.0,         1.0]
    ])

    poses = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])
        cam_pose = rot_z @ base_pose
        poses.append(cam_pose)

    return poses

def main():
    start = timeit.default_timer()
    # clear_output_dirs()
    step_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".STEP") or f.endswith(".step")]

    r = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    with open(METADATA_PATH, "w") as fj:
        for step_file in step_files:
            mesh = load_step_part(os.path.join(MODEL_DIR, step_file))
            part_name = os.path.splitext(step_file)[0]
            print(f"Rendering {part_name}")
                
            camera_poses = generate_orbit_camera_poses()

            for i, pose in enumerate(camera_poses):
                """
                img = render_single_view(mesh, pose, r)
                img_path = os.path.join(IMG_PATH, f"{part_name}_{i}.png")
                Image.fromarray(img).save(img_path)
                """

                json_line = {"image": f"images/{part_name}_{i}.png", "label": part_name}
                fj.write(json.dumps(json_line) + "\n")
    
    r.delete()
    stop = timeit.default_timer()

    print(f"Finished all stage 1 pyrenders in {(stop - start):.2f}s. Saved metadata to {METADATA_PATH}.")


if __name__ == "__main__":
    main()