#!/usr/bin/env python3
"""
Converts GARField tracking data to URDF with joint structure and bone visualization
"""

import json
import numpy as np
import trimesh
from pathlib import Path
from loguru import logger
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class RSRDToURDFConverter:
    """Convert RSRD GARField tracking data to URDF format with joint inference."""
    
    def __init__(self, track_dir: Path, output_dir: Path):
        self.track_dir = Path(track_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # motion data
        self.motion_data = None
        self.num_parts = 0
        self.num_frames = 0
        
        # joint structure
        self.joints = {}
        self.links = {}
        self.joint_hierarchy = {}
        
        logger.info(f"RSRD to URDF Converter initialized")
        logger.info(f"Track dir: {self.track_dir}")
        logger.info(f"Output dir: {self.output_dir}")
    
    def load_tracking_data(self) -> bool:
        """Load GARField tracking data from keyframes.txt"""
        
        keyframes_path = self.track_dir / "keyframes.txt"
        
        if not keyframes_path.exists():
            logger.error(f"Keyframes file not found: {keyframes_path}")
            return False
        
        try:
            with open(keyframes_path, 'r') as f:
                data = json.load(f)
            
            # Extract motion data: (time, parts, 7) - [qw, qx, qy, qz, x, y, z]
            self.motion_data = np.array(data['part_deltas'])
            self.num_frames, self.num_parts = self.motion_data.shape[:2]
            
            logger.info(f"Loaded tracking data:")
            logger.info(f"  {self.num_frames} frames")
            logger.info(f"  {self.num_parts} parts")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load tracking data: {e}")
            return False
    
    def analyze_motion_patterns(self) -> Dict:
        """Analyze motion patterns to infer joint types and relationships."""
        
        logger.info("Analyzing motion patterns for joint inference...")
        
        analysis = {
            'part_motion_types': {},
            'joint_candidates': [],
            'motion_stats': {}
        }
        
        for part_idx in range(self.num_parts):
            poses = self.motion_data[:, part_idx, :]
            
            # separate rotation and translation
            quaternions = poses[:, :4]  # [qw, qx, qy, qz]
            translations = poses[:, 4:7]  # [x, y, z]
            
            # analyze translation patterns
            translation_range = np.ptp(translations, axis=0)  # Range in each axis
            translation_std = np.std(translations, axis=0)    # Variability
            
            # analyze rotation patterns (convert quaternions to euler angles for analysis)
            rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])  # note-to-self: scipy wants [x,y,z,w]
            euler_angles = rotations.as_euler('xyz')
            rotation_range = np.ptp(euler_angles, axis=0)
            rotation_std = np.std(euler_angles, axis=0)
            
            # classify motion type
            motion_type = self._classify_motion_type(translation_range, rotation_range)
            
            analysis['part_motion_types'][part_idx] = {
                'type': motion_type,
                'translation_range': translation_range.tolist(),
                'rotation_range': rotation_range.tolist(),
                'translation_std': translation_std.tolist(),
                'rotation_std': rotation_std.tolist()
            }
            
            logger.info(f"  Part {part_idx}: {motion_type}")
            logger.info(f"    Translation range: {translation_range}")
            logger.info(f"    Rotation range: {rotation_range}")
        
        # infer joint relationships
        analysis['joint_candidates'] = self._infer_joint_relationships(analysis['part_motion_types'])
        
        return analysis
    
    def _classify_motion_type(self, trans_range: np.ndarray, rot_range: np.ndarray) -> str:
        """Classify the type of motion for a part."""
        
        # thresholds for motion detection
        TRANS_THRESHOLD = 0.01
        ROT_THRESHOLD = 0.1
        
        has_translation = np.any(trans_range > TRANS_THRESHOLD)
        has_rotation = np.any(rot_range > ROT_THRESHOLD)
        
        if not has_translation and not has_rotation:
            return "fixed"
        elif has_translation and not has_rotation:
            # check if translation is primarily along one axis
            dominant_axis = np.argmax(trans_range)
            if trans_range[dominant_axis] > 2 * np.sum(trans_range) / 3:
                return f"prismatic_{['x','y','z'][dominant_axis]}"
            else:
                return "floating"
        elif has_rotation and not has_translation:
            # check if rotation is primarily around one axis
            dominant_axis = np.argmax(rot_range)
            if rot_range[dominant_axis] > 2 * np.sum(rot_range) / 3:
                return f"revolute_{['x','y','z'][dominant_axis]}"
            else:
                return "floating"
        else:
            return "floating"
    
    def _infer_joint_relationships(self, motion_types: Dict) -> List[Dict]:
        """Infer parent-child relationships between parts based on motion."""
        
        joints = []
        
        # find the most static part as base
        static_parts = [idx for idx, info in motion_types.items() if info['type'] == 'fixed']
        
        if static_parts:
            base_part = static_parts[0]
        else:
            # choose part with least motion as base
            motion_scores = {}
            for idx, info in motion_types.items():
                score = np.sum(info['translation_range']) + np.sum(info['rotation_range'])
                motion_scores[idx] = score
            base_part = min(motion_scores, key=motion_scores.get)
        
        logger.info(f"  Base part: {base_part}")
        
        # create joints for all other parts
        for part_idx, motion_info in motion_types.items():
            if part_idx == base_part:
                continue
            
            motion_type = motion_info['type']
            
            if motion_type.startswith('revolute'):
                axis = motion_type.split('_')[1]
                joint_type = 'revolute'
                joint_axis = [1, 0, 0] if axis == 'x' else [0, 1, 0] if axis == 'y' else [0, 0, 1]
            elif motion_type.startswith('prismatic'):
                axis = motion_type.split('_')[1]
                joint_type = 'prismatic'
                joint_axis = [1, 0, 0] if axis == 'x' else [0, 1, 0] if axis == 'y' else [0, 0, 1]
            else:
                joint_type = 'fixed'
                joint_axis = [0, 0, 1]
            
            joints.append({
                'name': f'part_{base_part}_to_part_{part_idx}',
                'type': joint_type,
                'parent': f'part_{base_part}',
                'child': f'part_{part_idx}',
                'axis': joint_axis,
                'origin': [0, 0, 0], # may refine later
                'rpy': [0, 0, 0]
            })
        
        return joints
    
    def extract_part_meshes(self) -> Dict[str, Path]:
        """Extract mesh for each part (simplified approach using basic shapes)."""
        
        logger.info("🎨 Creating part meshes...")
        
        mesh_files = {}
        
        for part_idx in range(self.num_parts):
            # for now create a box mesh for each part
            mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
            
            # color based on part type
            colors = [
                [200, 200, 200, 255],  # Gray
                [50, 200, 50, 255],    # Green  
                [200, 50, 200, 255],   # Magenta
                [50, 50, 200, 255],    # Blue
                [200, 200, 50, 255],   # Yellow
                [200, 50, 50, 255],    # Red
            ]
            color = colors[part_idx % len(colors)]
            mesh.visual.vertex_colors = color
            
            mesh_path = self.output_dir / f"part_{part_idx:02d}.stl"
            mesh.export(str(mesh_path))
            
            mesh_files[f"part_{part_idx:02d}"] = mesh_path
            
            logger.info(f" Created mesh for part {part_idx}")
        
        return mesh_files
    
    def generate_urdf(self, mesh_files: Dict[str, Path], joint_candidates: List[Dict]) -> Path:
        """Generate URDF file with proper joint structure."""
        
        logger.info("Generating URDF...")
        
        # create URDF root
        robot = ET.Element("robot", name="garfield_object")
        
        # add base link
        base_link = ET.SubElement(robot, "link", name="base_link")
        base_visual = ET.SubElement(base_link, "visual")
        base_geom = ET.SubElement(base_visual, "geometry")
        ET.SubElement(base_geom, "box", size="0.001 0.001 0.001")  # Tiny invisible base
        
        # add part links
        for part_idx in range(self.num_parts):
            link_name = f"part_{part_idx:02d}"
            mesh_file = mesh_files[link_name]
            
            link = ET.SubElement(robot, "link", name=link_name)
            
            # visual
            visual = ET.SubElement(link, "visual")
            vis_geom = ET.SubElement(visual, "geometry")
            ET.SubElement(vis_geom, "mesh", filename=mesh_file.name, scale="1 1 1")
            
            # collision
            collision = ET.SubElement(link, "collision")
            col_geom = ET.SubElement(collision, "geometry")
            ET.SubElement(col_geom, "mesh", filename=mesh_file.name, scale="1 1 1")
            
            # inertial (simple)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.1")
            inertia = ET.SubElement(inertial, "inertia")
            inertia.set("ixx", "0.001")
            inertia.set("iyy", "0.001") 
            inertia.set("izz", "0.001")
            inertia.set("ixy", "0")
            inertia.set("ixz", "0")
            inertia.set("iyz", "0")
        
        # add joints
        for joint_info in joint_candidates:
            joint = ET.SubElement(robot, "joint", name=joint_info['name'], type=joint_info['type'])
            
            # parent/child
            ET.SubElement(joint, "parent", link=joint_info['parent'])
            ET.SubElement(joint, "child", link=joint_info['child'])
            
            # origin
            origin_xyz = " ".join(map(str, joint_info['origin']))
            origin_rpy = " ".join(map(str, joint_info['rpy']))
            ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
            
            # axis (for revolute/prismatic)
            if joint_info['type'] in ['revolute', 'prismatic']:
                axis_xyz = " ".join(map(str, joint_info['axis']))
                ET.SubElement(joint, "axis", xyz=axis_xyz)
                
                # Joint limits
                if joint_info['type'] == 'revolute':
                    ET.SubElement(joint, "limit", lower="-1.57", upper="1.57", effort="10", velocity="1")
                else:  # prismatic
                    ET.SubElement(joint, "limit", lower="-0.1", upper="0.1", effort="10", velocity="1")
        
        urdf_path = self.output_dir / "garfield_object.urdf"
        
        from xml.dom import minidom
        rough_string = ET.tostring(robot, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        
        pretty_lines = [line for line in pretty.split('\n') if line.strip()]
        pretty = '\n'.join(pretty_lines)
        
        with open(urdf_path, 'w') as f:
            f.write(pretty)
        
        logger.info(f" Generated URDF: {urdf_path}")
        return urdf_path
    
    def export_motion_data(self) -> List[Path]:
        """Export motion data as CSV files for visualization."""
        
        logger.info("Exporting motion data")
        
        pose_files = []
        
        for part_idx in range(self.num_parts):
            csv_path = self.output_dir / f"part_{part_idx:02d}_poses.csv"
            
            with open(csv_path, 'w') as f:
                f.write("timestep,x,y,z,qx,qy,qz,qw\n")
                
                for frame_idx in range(self.num_frames):
                    # Get pose: [qw, qx, qy, qz, x, y, z]
                    pose = self.motion_data[frame_idx, part_idx]
                    qw, qx, qy, qz, x, y, z = pose
                    
                    # Write as: timestep, x, y, z, qx, qy, qz, qw
                    f.write(f"{frame_idx},{x},{y},{z},{qx},{qy},{qz},{qw}\n")
            
            pose_files.append(csv_path)
            logger.info(f"Exported motion for part {part_idx}")
        
        return pose_files
    
    def create_motion_analysis_plot(self, analysis: Dict) -> Path:
        """Create visualization of motion analysis."""
        
        logger.info("Creating motion analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GARField Motion Analysis for URDF Generation')
        
        # 1. Translation ranges
        parts = list(range(self.num_parts))
        trans_ranges = []
        rot_ranges = []
        
        for part_idx in parts:
            info = analysis['part_motion_types'][part_idx]
            trans_ranges.append(info['translation_range'])
            rot_ranges.append(info['rotation_range'])
        
        trans_ranges = np.array(trans_ranges)
        rot_ranges = np.array(rot_ranges)
        
        # Translation plot
        axes[0, 0].bar(parts, trans_ranges[:, 0], alpha=0.7, label='X', color='red')
        axes[0, 0].bar(parts, trans_ranges[:, 1], alpha=0.7, label='Y', color='green', bottom=trans_ranges[:, 0])
        axes[0, 0].bar(parts, trans_ranges[:, 2], alpha=0.7, label='Z', color='blue', bottom=trans_ranges[:, 0] + trans_ranges[:, 1])
        axes[0, 0].set_xlabel('Part Index')
        axes[0, 0].set_ylabel('Translation Range (m)')
        axes[0, 0].set_title('Translation Ranges')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rotation plot
        axes[0, 1].bar(parts, rot_ranges[:, 0], alpha=0.7, label='Roll', color='red')
        axes[0, 1].bar(parts, rot_ranges[:, 1], alpha=0.7, label='Pitch', color='green', bottom=rot_ranges[:, 0])
        axes[0, 1].bar(parts, rot_ranges[:, 2], alpha=0.7, label='Yaw', color='blue', bottom=rot_ranges[:, 0] + rot_ranges[:, 1])
        axes[0, 1].set_xlabel('Part Index')
        axes[0, 1].set_ylabel('Rotation Range (rad)')
        axes[0, 1].set_title('Rotation Ranges')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Motion types
        motion_types = [analysis['part_motion_types'][i]['type'] for i in parts]
        type_colors = {'fixed': 'gray', 'revolute_x': 'red', 'revolute_y': 'green', 'revolute_z': 'blue',
                      'prismatic_x': 'orange', 'prismatic_y': 'purple', 'prismatic_z': 'cyan', 'floating': 'black'}
        colors = [type_colors.get(t, 'black') for t in motion_types]
        
        axes[1, 0].bar(parts, [1]*len(parts), color=colors)
        axes[1, 0].set_xlabel('Part Index')
        axes[1, 0].set_ylabel('Motion Type')
        axes[1, 0].set_title('Inferred Motion Types')
        axes[1, 0].set_yticks([0.5])
        axes[1, 0].set_yticklabels(['Type'])
        
        # Add legend for motion types
        for motion_type, color in type_colors.items():
            if motion_type in motion_types:
                axes[1, 0].bar([], [], color=color, label=motion_type)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Joint structure
        joint_info = analysis['joint_candidates']
        joint_names = [j['name'] for j in joint_info]
        joint_types = [j['type'] for j in joint_info]
        
        axes[1, 1].barh(range(len(joint_names)), [1]*len(joint_names), 
                       color=[type_colors.get(t, 'black') for t in joint_types])
        axes[1, 1].set_yticks(range(len(joint_names)))
        axes[1, 1].set_yticklabels([name.replace('part_', 'P') for name in joint_names])
        axes[1, 1].set_xlabel('Joint')
        axes[1, 1].set_title('Inferred Joint Structure')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "motion_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Motion analysis plot saved: {plot_path}")
        return plot_path
    
    def convert(self) -> bool:
        """Main conversion process."""
        
        logger.info("Starting RSRD to URDF conversion...")
        
        # 1. Load tracking data
        if not self.load_tracking_data():
            return False
        
        # 2. Analyze motion patterns
        analysis = self.analyze_motion_patterns()
        
        # 3. Extract/create part meshes
        mesh_files = self.extract_part_meshes()
        
        # 4. Generate URDF
        urdf_path = self.generate_urdf(mesh_files, analysis['joint_candidates'])
        
        # 5. Export motion data
        pose_files = self.export_motion_data()
        
        # 6. Create analysis plot
        plot_path = self.create_motion_analysis_plot(analysis)
        
        # Summary
        logger.info("---CONVERSION COMPLETE---")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"URDF file: {urdf_path.name}")
        logger.info(f"Parts: {len(mesh_files)}")
        logger.info(f"Joints: {len(analysis['joint_candidates'])}")
        logger.info(f"Motion files: {len(pose_files)}")
        logger.info(f"Analysis plot: {plot_path.name}")
        
        return True

def main():
    """Main function to run conversion."""
    
    track_dir = Path("/home/nick/rsrd/outputs/redbox/tracking")
    output_dir = Path("/home/nick/rsrd/garfield_urdf_output")
    
    converter = RSRDToURDFConverter(track_dir, output_dir)
    success = converter.convert()
    
    if success:
        logger.info("RSRD to URDF conversion successful!")
    else:
        logger.error("Conversion failed :(")

if __name__ == "__main__":
    main()
