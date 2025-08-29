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
    """Visualize bones and joints for GARField parts using actual point cloud data."""
    
    def __init__(self, server: viser.ViserServer, urdf_path: Path, motion_dir: Path, optimizer=None):
        self.server = server
        self.urdf_path = urdf_path
        self.motion_dir = motion_dir
        self.optimizer = optimizer  # RigidGroupOptimizer with actual NeRF data
        
        # URDF data
        self.joints = {}
        self.motion_data = {}
        self.num_frames = 0
        
        # Geometry analysis data
        self.part_geometry = {}  # PCA results for each part
        
        # Visualization handles
        self.bone_handles = {}
        self.joint_handles = {}
        
        # State flags
        self.bones_created = False
        self.bones_visible = False
        
        # Settings
        self.show_bones = False
        self.bone_thickness = 0.002  # Much thinner bones (2mm instead of 5mm)
        self.joint_radius = 0.003    # Smaller joint spheres (3mm instead of 8mm)
        
        self.load_data()
    
    def load_data(self):
        """Load URDF and motion data, then analyze part geometry."""
        self.parse_urdf()
        self.load_motion_data()
        if self.optimizer:
            self.analyze_part_geometry()
    
    def analyze_part_geometry(self):
        """Extract point clouds from NeRF and compute PCA for each part."""
        if not self.optimizer:
            logger.warning("No optimizer provided - cannot analyze part geometry")
            return
            
        logger.info("Analyzing part geometry from NeRF data...")
        
        dig_model = self.optimizer.dig_model
        all_means = dig_model.means.detach().cpu().numpy()
        
        for group_idx in range(len(self.optimizer.group_masks)):
            part_name = f"part_{group_idx:02d}"
            group_mask = self.optimizer.group_masks[group_idx].detach().cpu().numpy()
            
            # Extract points for this part
            part_points = all_means[group_mask]
            
            if len(part_points) < 3:
                logger.warning(f"Part {part_name} has too few points ({len(part_points)}) for PCA")
                continue
                
            # Compute centroid
            centroid = onp.mean(part_points, axis=0)
            
            # Center the points
            centered_points = part_points - centroid
            
            # Compute PCA
            cov_matrix = onp.cov(centered_points.T)
            eigenvalues, eigenvectors = onp.linalg.eig(cov_matrix)
            
            # Sort by eigenvalues (largest first) 
            idx = onp.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Store geometry analysis
            self.part_geometry[part_name] = {
                'centroid': centroid,
                'points': part_points,
                'num_points': len(part_points),
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'primary_axis': eigenvectors[:, 0],
                'extent_primary': onp.sqrt(eigenvalues[0]) * 2,
                'extent_secondary': onp.sqrt(eigenvalues[1]) * 2 if len(eigenvalues) > 1 else 0,
                'extent_tertiary': onp.sqrt(eigenvalues[2]) * 2 if len(eigenvalues) > 2 else 0,
            }
            
            logger.info(f"Part {part_name}: {len(part_points)} points, "
                       f"primary extent: {onp.sqrt(eigenvalues[0]) * 2:.3f}, "
                       f"primary axis: {eigenvectors[:, 0]}")
        
        logger.info(f"Analyzed geometry for {len(self.part_geometry)} parts")
    
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
        
        logger.info(f"Loaded motion data: {self.num_frames} frames")
    
    def create_bone_geometry(self, start_pos: onp.ndarray, end_pos: onp.ndarray) -> trimesh.Trimesh:
        """Create octahedral bone geometry between two points, similar to Blender's bone display."""
        direction = end_pos - start_pos
        length = onp.linalg.norm(direction)
        
        if length < 1e-6:
            return trimesh.creation.icosphere(radius=self.bone_thickness*3)
        
        # Create octahedral bone similar to Blender's display
        # Scale factor for the bone width relative to length
        width_scale = 0.15
        bone_width = max(length * width_scale, self.bone_thickness * 2)
        
        # Create simplified diamond/octahedral shape
        # More stable geometry with proper face winding
        vertices = onp.array([
            [0.0, 0.0, 0.0],                    # tail point (start)
            [bone_width, 0.0, length*0.3],     # +X at 30% 
            [0.0, bone_width, length*0.3],     # +Y at 30%
            [-bone_width, 0.0, length*0.3],    # -X at 30%
            [0.0, -bone_width, length*0.3],    # -Y at 30%
            [bone_width*0.5, 0.0, length*0.7], # +X at 70% (smaller)
            [0.0, bone_width*0.5, length*0.7], # +Y at 70% (smaller)
            [-bone_width*0.5, 0.0, length*0.7], # -X at 70% (smaller)
            [0.0, -bone_width*0.5, length*0.7], # -Y at 70% (smaller)
            [0.0, 0.0, length]                 # head point (end)
        ])
        
        # Define faces with proper winding order (counter-clockwise)
        faces = onp.array([
            # From tail to middle ring (4 triangles)
            [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
            # Middle section (8 triangles connecting two rings)
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
            [3, 4, 8], [3, 8, 7], [4, 1, 5], [4, 5, 8],
            # From smaller ring to head (4 triangles)
            [5, 6, 9], [6, 7, 9], [7, 8, 9], [8, 5, 9]
        ])
        
        # Create the mesh
        try:
            bone = trimesh.Trimesh(vertices=vertices, faces=faces, validate=True)
            # Fix any mesh issues
            bone.fix_normals()
        except Exception as e:
            logger.warning(f"Failed to create octahedral bone, falling back to cylinder: {e}")
            # Fallback to simple cylinder if octahedron fails
            bone = trimesh.creation.cylinder(radius=self.bone_thickness, height=length)
        
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
        
        # Position bone at start position
        bone.apply_translation(start_pos)
        
        return bone

    def create_octahedral_bone(self, center: onp.ndarray, axis: onp.ndarray, length: float) -> trimesh.Trimesh:
        """Create an octahedral bone at a center point along a given axis."""
        # Create bone extending from center - length/2 to center + length/2 along axis
        axis_norm = axis / (onp.linalg.norm(axis) + 1e-8)
        start_pos = center - axis_norm * length * 0.5
        end_pos = center + axis_norm * length * 0.5
        return self.create_bone_geometry(start_pos, end_pos)
    
    def create_bones_once(self):
        """Create bone geometry based on PCA analysis of part geometry."""
        if not self.part_geometry:
            logger.warning("No part geometry data available - cannot create bones")
            return
            
        logger.info("Creating bones based on PCA analysis of NeRF parts")
        
        bone_count = 0
        
        # Create bones inside each part along their principal axes
        for part_name, geometry in self.part_geometry.items():
            primary_axis = geometry['primary_axis']
            extent = geometry['extent_primary']
            
            # Create bone along primary axis, centered at ORIGIN (not absolute NeRF centroid)
            # The bone will be positioned correctly during update_bones() using part_deltas
            bone_length = min(extent * 0.3, 0.03)  # Even smaller bones: 30% of extent, max 3cm
            origin = onp.array([0.0, 0.0, 0.0])  # Start at origin
            bone_start = origin - (primary_axis * bone_length / 2)
            bone_end = origin + (primary_axis * bone_length / 2)
            
            # Create octahedral bone geometry
            bone_mesh = self.create_bone_geometry(bone_start, bone_end)
            
            # Set random color on the mesh itself
            color = onp.random.rand(3)
            bone_mesh.visual.vertex_colors = (color[0]*255, color[1]*255, color[2]*255, 255)
            
            # Add to viser scene
            bone_handle = self.server.scene.add_mesh_trimesh(
                f"/object/bones/{part_name}",
                bone_mesh
            )
            
            self.bone_handles[part_name] = bone_handle
            bone_count += 1
            
            logger.info(f"Created bone for {part_name}: length={bone_length:.3f}, "
                       f"axis={primary_axis}, positioned at origin (will be transformed during updates)")
            
        # Create joint spheres at part connections
        self.create_joint_spheres()
        
        # Mark bones as created
        self.bones_created = True
        
        logger.info(f"Created {bone_count} bones and joint spheres")
    
    def create_joint_spheres(self):
        """Create spherical joints where parts connect."""
        if not self.joints or not self.part_geometry:
            return
            
        # Use first frame to get part positions
        first_frame = min(self.motion_data.keys()) if self.motion_data else 0
        
        for joint_name, joint_info in self.joints.items():
            parent_link = joint_info['parent']
            child_link = joint_info['child']
            
            # Convert link names to part names
            parent_part = parent_link.replace('_link', '')
            child_part = child_link.replace('_link', '')
            
            # Fix naming convention (part_0 -> part_00)
            if parent_part.startswith('part_') and len(parent_part) == 6:
                part_num = int(parent_part[-1])
                parent_part = f"part_{part_num:02d}"
            if child_part.startswith('part_') and len(child_part) == 6:
                part_num = int(child_part[-1])
                child_part = f"part_{part_num:02d}"
                
            # Get part centroids (relative to origin, not absolute NeRF coordinates)
            if parent_part in self.part_geometry and child_part in self.part_geometry:
                # Place joint sphere at origin for now - will be positioned during updates
                joint_position = onp.array([0.0, 0.0, 0.0])
                
                # Create joint sphere
                joint_sphere = trimesh.creation.icosphere(radius=self.joint_radius)
                joint_sphere.apply_translation(joint_position)
                
                # Set orange color on the mesh
                joint_sphere.visual.vertex_colors = (255, 127, 0, 255)  # Orange
                
                # Add to viser scene
                joint_handle = self.server.scene.add_mesh_trimesh(
                    f"/object/joints/{joint_name}",
                    joint_sphere
                )
                
                self.joint_handles[joint_name] = joint_handle
                
                logger.info(f"Created joint sphere for {joint_name} at origin (will be positioned during updates)")

    def update_bones(self, tstep: int, part_deltas, scale: float = 1.0):
        """Update bone positions and orientations for the current frame"""
        if not self.bones_created or not self.show_bones:
            return
            
        try:
            # Create a list of items to avoid "dictionary changed size during iteration"
            bone_items = list(self.bone_handles.items())
            for part_name, bone_handle in bone_items:
                # Convert part names from part_0 format to part_00 format
                original_part_name = part_name.replace("_bone", "")
                if len(original_part_name.split("_")) == 2 and len(original_part_name.split("_")[1]) == 1:
                    part_idx_str = original_part_name.split("_")[1]
                    formatted_part_name = f"part_{part_idx_str.zfill(2)}"
                else:
                    formatted_part_name = original_part_name
                
                # Get part index
                part_idx = int(formatted_part_name.split("_")[1])
                
                # Get current part transformation
                if part_idx < len(part_deltas):
                    current_delta = part_deltas[part_idx]
                    delta_values = current_delta.detach().cpu().numpy().flatten()
                    se3_transform = vtf.SE3(delta_values)
                    
                    # Get original geometry data for this part
                    if formatted_part_name in self.part_geometry:
                        original_axis = self.part_geometry[formatted_part_name]['primary_axis']
                        
                        # Apply SE3 transformation to get current part position
                        current_translation = se3_transform.translation() * scale
                        current_rotation_matrix = se3_transform.rotation().as_matrix()
                        
                        # Transform the axis using current part rotation
                        transformed_axis = current_rotation_matrix @ original_axis
                        
                        # Position bone at the current part position (not offset from NeRF centroid)
                        transformed_centroid = current_translation
                        
                        # Create rotation matrix to align bone with transformed axis
                        z_axis = transformed_axis / onp.linalg.norm(transformed_axis)
                        x_axis = onp.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else onp.array([0, 1, 0])
                        x_axis = x_axis - onp.dot(x_axis, z_axis) * z_axis
                        x_axis = x_axis / onp.linalg.norm(x_axis)
                        y_axis = onp.cross(z_axis, x_axis)
                        rotation_matrix = onp.column_stack([x_axis, y_axis, z_axis])
                        
                        # Update bone handle properties directly (proper viser API)
                        bone_handle.position = transformed_centroid
                        bone_handle.rotation = vtf.SO3.from_matrix(rotation_matrix)
                        
        except Exception as e:
            logger.warning(f"Failed to update bones: {e}")

        # Update joint spheres using proper viser handle properties
        try:
            # Create a list of items to avoid "dictionary changed size during iteration"
            joint_items = list(self.joint_handles.items())
            for joint_name, joint_handle in joint_items:
                # Parse joint name to get part indices
                parts = joint_name.replace("part_", "").split("_to_part_")
                if len(parts) == 2:
                    part1_idx, part2_idx = int(parts[0]), int(parts[1])
                    
                    # Get transformations for both parts
                    if part1_idx < len(part_deltas) and part2_idx < len(part_deltas):
                        delta1 = part_deltas[part1_idx].detach().cpu().numpy().flatten()
                        delta2 = part_deltas[part2_idx].detach().cpu().numpy().flatten()
                        se3_1 = vtf.SE3(delta1)
                        se3_2 = vtf.SE3(delta2)
                        
                        # Get current part positions (not offset from NeRF centroids)
                        part1_name = f"part_{part1_idx:02d}"
                        part2_name = f"part_{part2_idx:02d}"
                        
                        # Apply transformations to get current positions directly
                        new_pos1 = se3_1.translation() * scale
                        new_pos2 = se3_2.translation() * scale
                        
                        # Joint position is midpoint between connected parts
                        joint_position = (new_pos1 + new_pos2) / 2.0
                        
                        # Update joint handle position directly (proper viser API)
                        joint_handle.position = joint_position
                    
        except Exception as e:
            logger.warning(f"Failed to update joints: {e}")

    def toggle_bones(self, show: bool):
        """Toggle bone and joint visibility."""
        self.show_bones = show
        if not show:
            # Hide all bones
            for part_name in self.bone_handles.keys():
                try:
                    self.bone_handles[part_name].visible = False
                except:
                    pass
            
            # Hide all joints
            for joint_name in self.joint_handles.keys():
                try:
                    self.joint_handles[joint_name].visible = False
                except:
                    pass
            
            logger.info("Bones and joints hidden")
        else:
            # Create bones if this is the first time
            if not self.bone_handles:
                self.create_bones_once()
            else:
                # Show existing bones
                for part_name in self.bone_handles.keys():
                    try:
                        self.bone_handles[part_name].visible = True
                    except:
                        pass
                
                # Show existing joints
                for joint_name in self.joint_handles.keys():
                    try:
                        self.joint_handles[joint_name].visible = True
                    except:
                        pass
            
            logger.info("Bones and joints enabled")


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

    # Initialize bone visualizer with optimizer for PCA analysis
    urdf_path = Path("/home/nick/rsrd/garfield_urdf_output/garfield_object.urdf")
    motion_dir = Path("/home/nick/rsrd/garfield_urdf_output")
    bone_viz = BoneVisualizer(server, urdf_path, motion_dir, optimizer)

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
    with server.gui.add_folder("Joint/Bone Visualization"):
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
                save_status.value = f"Saved {list_traj.shape[0]} trajectories!"
            else:
                save_status.value = "Failed to save trajectories"
        else:
            save_status.value = "No trajectories to save"

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
                save_status.value = f"Loaded {list_traj.shape[0]} trajectories from {info.get('timestamp', 'previous session')}"
                
                logger.info(f"Loaded {list_traj.shape[0]} trajectories successfully")
                
            else:
                save_status.value = "No saved trajectories found"
                logger.warning("No trajectories found in save directory")
                
        except Exception as e:
            save_status.value = f"Failed to load: {str(e)[:50]}..."
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
            current_part_deltas = optimizer.part_deltas[tstep]
            bone_viz.update_bones(tstep, current_part_deltas, scale=(1 / optimizer.dataset_scale))

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
            bone_viz.update_bones(tstep, part_deltas, scale=(1 / optimizer.dataset_scale))

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
