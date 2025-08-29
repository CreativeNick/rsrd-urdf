"""
Run object-centric robot motion planning. (robot do! haha).

now with joint/bone visualization! (wip)
"""

# JAX by default pre-allocates 75%, which will cause OOM with nerfstudio model loading.
# This line needs to go above any jax import.
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import json
from typing import Literal, Optional
from threading import Lock

import jaxlie
import time
from pathlib import Path
import moviepy.editor as mpy

import cv2
import numpy as onp
import jax.numpy as jnp
import torch
import warp as wp

import viser
import viser.extras
import viser.transforms as vtf
import tyro
from loguru import logger

from nerfstudio.utils.eval_utils import eval_setup
from jaxmp.extras.urdf_loader import load_urdf

from rsrd.extras.cam_helpers import get_ns_camera_at_origin
from rsrd.robot.motion_plan_yumi import YUMI_REST_POSE
from rsrd.extras.zed import Zed
from rsrd.motion.motion_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
from rsrd.motion.atap_loss import ATAPConfig
from rsrd.extras.viser_rsrd import ViserRSRD
from rsrd.robot.planner import PartMotionPlanner
from rsrd.robot.trajectory_saver import create_trajectory_saver
import rsrd.transforms as tf
from autolab_core import RigidTransform

# Additional imports for bone visualization
import trimesh
import xml.etree.ElementTree as ET


class BoneVisualizer:
    """Add bone/joint visualization to the planner."""
    
    def __init__(self, server: viser.ViserServer, urdf_path: Path, motion_dir: Path):
        self.server = server
        self.urdf_path = urdf_path
        self.motion_dir = motion_dir
        
        # URDF data
        self.joints = {}
        self.motion_data = {}
        self.num_frames = 0
        
        # Visualization handles
        self.bone_handles = {}
        
        # Settings
        self.show_bones = False
        self.bone_thickness = 0.005
        
        self.load_data()
    
    def load_data(self):
        """Load URDF and motion data."""
        self.parse_urdf()
        self.load_motion_data()
    
    def parse_urdf(self):
        """Parse URDF for joint information."""
        if not self.urdf_path.exists():
            logger.warning(f"URDF not found: {self.urdf_path}")
            return
        
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            
            for joint_elem in root.findall('joint'):
                joint_name = joint_elem.get('name')
                joint_type = joint_elem.get('type')
                parent_link = joint_elem.find('parent').get('link')
                child_link = joint_elem.find('child').get('link')
                
                self.joints[joint_name] = {
                    'name': joint_name,
                    'type': joint_type,
                    'parent': parent_link,
                    'child': child_link,
                }
            
            logger.info(f"Loaded {len(self.joints)} joints from URDF")
            
        except Exception as e:
            logger.error(f"Failed to parse URDF: {e}")
    
    def load_motion_data(self):
        """Load motion data from CSV files."""
        pose_files = list(self.motion_dir.glob("part_*_poses.csv"))
        if not pose_files:
            logger.warning("No motion data found")
            return
        
        for pose_file in pose_files:
            part_name = pose_file.stem.replace('_poses', '')
            
            with open(pose_file, 'r') as f:
                lines = f.readlines()[1:]
                
                for line in lines:
                    parts = line.strip().split(',')
                    frame = int(parts[0])
                    pose = [float(x) for x in parts[1:]]
                    
                    if frame not in self.motion_data:
                        self.motion_data[frame] = {}
                    self.motion_data[frame][part_name] = pose
                    self.num_frames = max(self.num_frames, frame + 1)
        
        logger.info(f"🦴 Loaded motion data: {self.num_frames} frames")
    
    def create_bone_geometry(self, start_pos: onp.ndarray, end_pos: onp.ndarray) -> trimesh.Trimesh:
        """Create bone geometry between two points."""
        direction = end_pos - start_pos
        length = onp.linalg.norm(direction)
        
        if length < 1e-6:
            return trimesh.creation.icosphere(radius=self.bone_thickness*3)
        
        # Create bone with visible ends
        bone = trimesh.creation.cylinder(radius=self.bone_thickness, height=length)
        
        end1 = trimesh.creation.icosphere(radius=self.bone_thickness*2)
        end1.apply_translation([0, 0, -length/2])
        
        end2 = trimesh.creation.icosphere(radius=self.bone_thickness*2)
        end2.apply_translation([0, 0, length/2])
        
        bone = trimesh.util.concatenate([bone, end1, end2])
        
        # Align with direction
        if length > 1e-6:
            z_axis = onp.array([0, 0, 1])
            direction_norm = direction / length
            
            rotation_axis = onp.cross(z_axis, direction_norm)
            rotation_angle = onp.arccos(onp.clip(onp.dot(z_axis, direction_norm), -1, 1))
            
            if onp.linalg.norm(rotation_axis) > 1e-6:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    rotation_angle, rotation_axis
                )
                bone.apply_transform(rotation_matrix)
        
        # Position bone
        center = (start_pos + end_pos) / 2
        bone.apply_translation(center)
        
        return bone
    
    def update_bones(self, frame: int, scale: float = 1.0):
        """Update bone visualization for given frame."""
        if frame not in self.motion_data or not self.show_bones:
            # Clear bones if not showing
            for joint_name in list(self.bone_handles.keys()):
                try:
                    self.server.scene.remove(f"/object/bones/{joint_name}")
                except:
                    pass
            self.bone_handles.clear()
            return
        
        # Get part positions for this frame
        part_positions = {}
        for part_name, pose in self.motion_data[frame].items():
            position = onp.array(pose[:3]) * scale
            part_positions[part_name] = position
        
        # Update/create bones
        for joint_name, joint_info in self.joints.items():
            parent_link = joint_info['parent']
            child_link = joint_info['child']
            
            parent_part = parent_link.replace('_link', '')
            child_part = child_link.replace('_link', '')
            
            # Get positions
            if parent_part in part_positions:
                parent_pos = part_positions[parent_part]
            else:
                parent_pos = onp.array([0, 0, 0])
            
            if child_part in part_positions:
                child_pos = part_positions[child_part]
            else:
                continue
            
            # Create bone
            bone_mesh = self.create_bone_geometry(parent_pos, child_pos)
            bone_mesh.visual.vertex_colors = [255, 50, 50, 200]  # Bright red
            
            # Remove old bone
            bone_path = f"/object/bones/{joint_name}"
            if joint_name in self.bone_handles:
                try:
                    self.server.scene.remove(bone_path)
                except:
                    pass
            
            # Add new bone
            bone_handle = self.server.scene.add_mesh_trimesh(
                bone_path,
                bone_mesh,
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0)
            )
            self.bone_handles[joint_name] = bone_handle
    
    def toggle_bones(self, show: bool):
        """Toggle bone visibility."""
        self.show_bones = show
        if not show:
            # Clear all bones
            for joint_name in list(self.bone_handles.keys()):
                try:
                    self.server.scene.remove(f"/object/bones/{joint_name}")
                except:
                    pass
            self.bone_handles.clear()
            logger.info("[X] Bones hidden")
        else:
            logger.info("[O] Bones enabled")


def main(
    hand_mode: Literal["single", "bimanual"],
    track_dir: Path,
    zed_video_path: Optional[Path] = None,
):
    optimizer = get_optimizer(track_dir)
    logger.info("Initialized tracker.")

    server = viser.ViserServer()

    # Load URDF.
    urdf = load_urdf(
        robot_urdf_path=Path(__file__).parent
        / "data/yumi_description/urdf/yumi.urdf"
    )
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/yumi")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
    planner = PartMotionPlanner(optimizer, urdf)
    viser_urdf.update_cfg(onp.array(YUMI_REST_POSE))

    # Load camera position.
    T_cam_world = vtf.SE3.from_matrix(
        RigidTransform.load(
            Path(__file__).parent / "data/zed/zed_to_world.tf"
        ).matrix
    )
    server.scene.add_frame(
        "/camera",
        position=T_cam_world.translation(),
        wxyz=T_cam_world.rotation().wxyz,
        show_axes=False,
    )

    # Load object position, if available.
    if zed_video_path is not None:
        T_obj_world, (points, colors) = get_T_obj_world_from_zed(
            optimizer, zed_video_path, T_cam_world, track_dir
        )
        server.scene.add_point_cloud(
            "/camera/points",
            points=points,
            colors=colors,
            point_size=0.005,
        )
        free_object = False  # Fix at initialization zed point.
        
        if hand_mode == "bimanual":
            traj_generator = planner.plan_bimanual(T_obj_world)
        else:
            traj_generator = planner.plan_single(T_obj_world)

    else:
        T_obj_world = jaxlie.SE3.from_translation(jnp.array([0.4, 0.0, 0.0]))
        free_object = True
        traj_generator = None

    obj_frame_handle = server.scene.add_transform_controls(
        "/object",
        position=onp.array(T_obj_world.translation().squeeze()),
        wxyz=onp.array(T_obj_world.rotation().wxyz.squeeze()),
        active_axes=(free_object, free_object, free_object),
        scale=0.2,
    )

    # Load RSRD object.
    viser_rsrd = ViserRSRD(
        server,
        optimizer,
        root_node_name="/object",
        scale=(1 / optimizer.dataset_scale),
        show_hands=False,
    )
    initial_part_deltas = optimizer.part_deltas[0]
    viser_rsrd.update_cfg(initial_part_deltas)

    # Initialize bone visualizer
    urdf_path = Path("/home/nick/rsrd/garfield_urdf_output/garfield_object.urdf")
    motion_dir = Path("/home/nick/rsrd/garfield_urdf_output")
    bone_viz = BoneVisualizer(server, urdf_path, motion_dir)

    timesteps = optimizer.part_deltas.shape[0]
    move_obj_handler = server.gui.add_checkbox("Move object", free_object, disabled=(not free_object))
    generate_traj_handler = server.gui.add_button("Generate trajectory", disabled=free_object)
    traj_list_handler = server.gui.add_slider("trajectory", 0, 1, 1, 0, disabled=True)
    
    # Add trajectory saving functionality
    with server.gui.add_folder("💾 Trajectory Management"):
        save_traj_button = server.gui.add_button("Save trajectories", disabled=True)
        load_traj_button = server.gui.add_button("Load trajectories", disabled=False)
        save_status = server.gui.add_text("Status", initial_value="No trajectories to save")
    
    # Add bone visualization controls
    with server.gui.add_folder("🦴 Joint/Bone Visualization"):
        show_bones_checkbox = server.gui.add_checkbox("Show Bones", False)
        bone_thickness_slider = server.gui.add_slider("Bone Thickness", 0.001, 0.02, 0.001, 0.005)
        
        bone_info = server.gui.add_text(
            "URDF Info",
            initial_value=f"Joints: {len(bone_viz.joints)}, Motion Frames: {bone_viz.num_frames}"
        )
    
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)
    traj_gen_lock = Lock()
    
    # Create trajectory saver
    trajectory_saver = create_trajectory_saver(track_dir)
    
    # Check if saved trajectories exist and update load button status
    def check_existing_trajectories():
        try:
            existing_data = trajectory_saver.load_trajectories()
            if existing_data and 'all_trajectories' in existing_data and existing_data['all_trajectories'] is not None:
                n_traj = existing_data['info']['n_trajectories']
                save_status.value = f"Found {n_traj} saved trajectories - click Load to restore"
                return True
        except:
            pass
        save_status.value = "No saved trajectories found"
        return False
    
    has_existing_trajectories = check_existing_trajectories()
    
    # Check if there are existing trajectories to load
    traj_summary = trajectory_saver.get_trajectory_summary()
    if traj_summary.get('has_data', False):
        save_status.value = f"Found {traj_summary['n_trajectories']} saved trajectories from {traj_summary.get('timestamp', 'previous session')[:10]}"
    else:
        save_status.value = "No saved trajectories found"
        load_traj_button.disabled = True

    list_traj = jnp.array(YUMI_REST_POSE).reshape(1, 1, -1).repeat(timesteps, axis=1)

    # While moving object / fixing object, trajectories need to be regenerated.
    # So it should clear all the cache.
    @move_obj_handler.on_update
    def _(_):
        nonlocal list_traj, traj_generator
        generate_traj_handler.disabled = move_obj_handler.value
        curr_free_axes = move_obj_handler.value
        obj_frame_handle.active_axes = (curr_free_axes, curr_free_axes, curr_free_axes)

        traj_list_handler.value = 0
        traj_list_handler.disabled = True
        save_traj_button.disabled = True
        save_status.value = "No trajectories to save"
        if list_traj is not None:
            list_traj = jnp.array(YUMI_REST_POSE).reshape(1, 1, -1).repeat(timesteps, axis=1)

        T_obj_world = jaxlie.SE3(
            jnp.array([*obj_frame_handle.wxyz, *obj_frame_handle.position])
        )
        if hand_mode == "bimanual":
            traj_generator = planner.plan_bimanual(T_obj_world)
        else:
            traj_generator = planner.plan_single(T_obj_world)

    @generate_traj_handler.on_click
    def _(_):
        nonlocal list_traj
        assert traj_generator is not None
        generate_traj_handler.disabled = True
        move_obj_handler.disabled = True
        with traj_gen_lock:
            list_traj = next(traj_generator)
        generate_traj_handler.disabled = False
        traj_list_handler.max = list_traj.shape[0] - 1
        traj_list_handler.value = 0
        traj_list_handler.disabled = False
        move_obj_handler.disabled = False
        
        # Enable trajectory saving
        save_traj_button.disabled = False
        save_status.value = f"Ready to save {list_traj.shape[0]} trajectories"

    @save_traj_button.on_click
    def _(_):
        nonlocal list_traj
        if list_traj is not None and list_traj.shape[0] > 0:
            save_status.value = "Saving trajectories..."
            
            # Gather metadata
            metadata = {
                "hand_mode": hand_mode,
                "object_position": obj_frame_handle.position.tolist(),
                "object_orientation": obj_frame_handle.wxyz.tolist(),
                "selected_trajectory_idx": traj_list_handler.value,
                "total_timesteps": timesteps,
            }
            
            # Save trajectories using the separate saver module
            success = trajectory_saver.save_trajectories(
                trajectories=list_traj,
                kin_tree=planner.kin,
                metadata=metadata
            )
            
            if success:
                save_status.value = f"✅ Saved {list_traj.shape[0]} trajectories!"
            else:
                save_status.value = "❌ Failed to save trajectories"
        else:
            save_status.value = "❌ No trajectories to save"

    @load_traj_button.on_click
    def _(_):
        nonlocal list_traj
        save_status.value = "Loading trajectories..."
        
        try:
            # Load trajectories using the saver module
            loaded_data = trajectory_saver.load_trajectories()
            
            if loaded_data and 'all_trajectories' in loaded_data and loaded_data['all_trajectories'] is not None:
                loaded_traj = loaded_data['all_trajectories']
                info = loaded_data['info']
                
                # Convert to jax array and set as current trajectories
                list_traj = jnp.array(loaded_traj)
                
                # Update UI controls
                traj_list_handler.max = list_traj.shape[0] - 1
                traj_list_handler.value = info.get('selected_trajectory_idx', 0)
                traj_list_handler.disabled = False
                
                # Enable saving (since we now have trajectories)
                save_traj_button.disabled = False
                
                # Update status
                save_status.value = f"✅ Loaded {list_traj.shape[0]} trajectories from {info.get('timestamp', 'previous session')}"
                
                logger.info(f"Loaded {list_traj.shape[0]} trajectories successfully")
                
            else:
                save_status.value = "❌ No saved trajectories found"
                logger.warning("No trajectories found in save directory")
                
        except Exception as e:
            save_status.value = f"❌ Failed to load: {str(e)[:50]}..."
            logger.error(f"Failed to load trajectories: {e}")

    # Bone visualization callbacks
    @show_bones_checkbox.on_update
    def _(event):
        bone_viz.toggle_bones(event.target.value)
        
    @bone_thickness_slider.on_update
    def _(event):
        bone_viz.bone_thickness = event.target.value
        if bone_viz.show_bones:
            # Update bones with new thickness
            tstep = track_slider.value
            bone_viz.update_bones(tstep, scale=(1 / optimizer.dataset_scale))

    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps

        traj = list_traj[traj_list_handler.value]
        tstep = track_slider.value

        part_deltas = optimizer.part_deltas[tstep]
        viser_rsrd.update_cfg(part_deltas)
        viser_rsrd.update_hands(tstep)
        viser_urdf.update_cfg(onp.array(traj[tstep]))
        
        # Update bones if enabled
        if bone_viz.show_bones:
            bone_viz.update_bones(tstep, scale=(1 / optimizer.dataset_scale))

        time.sleep(0.05)


def get_optimizer(
    track_dir: Path,
) -> RigidGroupOptimizer:
    # Save the paths to the cache file.
    track_cache_path = track_dir / "cache_info.json"
    assert track_cache_path.exists()
    cache_data = json.loads(track_cache_path.read_text())
    is_obj_jointed = bool(cache_data["is_obj_jointed"])
    dig_config_path = Path(cache_data["dig_config_path"])
    track_data_path = track_dir / "keyframes.txt"

    # Load DIG model, create viewer.
    _, pipeline, _, _ = eval_setup(dig_config_path)
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    is_obj_jointed = False  # Unused anyway, for registration.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        ),
        altitude_down=0.0,
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
    )
    # Load keyframes.
    optimizer.load_tracks(track_data_path)
    hands = optimizer.hands_info
    assert hands is not None

    return optimizer


def get_T_obj_world_from_zed(
    optimizer: RigidGroupOptimizer,
    zed_video_path: Path,
    T_cam_world: vtf.SE3,
    track_dir: Path,
) -> tuple[jaxlie.SE3, tuple[onp.ndarray, onp.ndarray]]:
    """
    Get T_obj_world by registering the object in the scene.
    The ZED video shows the static scene with the object.
    """
    # Optimize, based on zed video.
    zed = Zed(str(zed_video_path))
    left, _, depth = zed.get_frame(depth=True)
    assert left is not None and depth is not None
    points, colors = zed.project_depth(
        left, depth, torch.Tensor(zed.get_K()).cuda(), subsample=8
    )

    # Optimize object pose.
    left_uint8 = (left.cpu().numpy() * 255).astype(onp.uint8)
    camera = get_ns_camera_at_origin(K=zed.get_K(), width=zed.width, height=zed.height)
    first_obs = optimizer.create_observation_from_rgb_and_camera(
        left_uint8, camera, metric_depth=depth.cpu().numpy()
    )
    renders = optimizer.initialize_obj_pose(first_obs, render=True, use_depth=True)
    _renders = []
    for r in renders:
        _left = cv2.resize(left_uint8, (r.shape[1], r.shape[0]))
        _left = cv2.cvtColor(_left, cv2.COLOR_BGR2RGB)
        _renders.append(r * 0.8 + _left * 0.2)
    out_clip = mpy.ImageSequenceClip(_renders, fps=30)
    out_clip.write_videofile(str(track_dir / "zed_registration.mp4"))

    T_obj_cam = optimizer.T_objreg_world
    # Convert opengl -> opencv
    T_obj_cam = (
        tf.SE3.from_rotation(tf.SO3.from_x_radians(torch.Tensor([torch.pi]).cuda()))
        @ T_obj_cam
    )
    # Put to world scale; tracking is performed + saved in nerfstudio coordinates.
    T_obj_cam = tf.SE3.from_rotation_and_translation(
        rotation=T_obj_cam.rotation(),
        translation=T_obj_cam.translation() / optimizer.dataset_scale,
    )

    # Optimize robot trajectory.
    T_obj_world = jaxlie.SE3(jnp.array(T_cam_world.wxyz_xyz)) @ jaxlie.SE3(
        jnp.array(T_obj_cam.wxyz_xyz.detach().cpu())
    )

    return T_obj_world, (points, colors)


if __name__ == "__main__":
    tyro.cli(main)
