"""
Trajectory saving utilities for RSRD robot planner.
Separate module to avoid interfering with core planner logic.
"""

import json
import numpy as onp
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
import jax.numpy as jnp
import jax.numpy as jnp


class TrajectorySaver:
    """Handles saving robot trajectories in multiple formats."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_trajectories(
        self,
        trajectories: jnp.ndarray,
        selected_idx: int,
        joint_names: list,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Save trajectories in multiple formats.
        
        Args:
            trajectories: Shape (n_trajectories, timesteps, n_joints)
            selected_idx: Index of currently selected trajectory
            joint_names: List of robot joint names
            metadata: Optional additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure we have valid data
            if trajectories is None or trajectories.shape[0] == 0:
                logger.warning("No trajectories to save")
                return False
                
            n_traj, timesteps, n_joints = trajectories.shape
            
            # Format 1: Raw numpy array (all trajectories)
            raw_path = self.save_dir / "raw_trajectories.npy"
            onp.save(raw_path, onp.array(trajectories))
            
            # Format 2: Selected trajectory only
            if 0 <= selected_idx < n_traj:
                selected_traj = trajectories[selected_idx]
                selected_path = self.save_dir / "selected_trajectory.npy"
                onp.save(selected_path, onp.array(selected_traj))
            else:
                logger.warning(f"Invalid selected_idx {selected_idx}, skipping selected trajectory save")
                selected_traj = None
            
            # Format 3: Metadata and info in JSON
            info_data = {
                "timestamp": str(onp.datetime64('now')),
                "n_trajectories": int(n_traj),
                "timesteps": int(timesteps),
                "n_joints": int(n_joints),
                "selected_trajectory_idx": int(selected_idx) if 0 <= selected_idx < n_traj else None,
                "joint_names": joint_names,
                "files_saved": [
                    "raw_trajectories.npy",
                    "selected_trajectory.npy" if selected_traj is not None else None,
                    "trajectory_info.json"
                ],
                "trajectory_stats": {
                    "max_joint_angle": float(onp.max(onp.abs(trajectories))),
                    "avg_joint_range": float(onp.mean(onp.max(trajectories, axis=1) - onp.min(trajectories, axis=1))),
                }
            }
            
            # Add user metadata if provided
            if metadata:
                info_data["user_metadata"] = metadata
            
            # Save selected trajectory data in JSON for readability
            if selected_traj is not None:
                info_data["selected_trajectory_sample"] = {
                    "first_pose": selected_traj[0].tolist(),
                    "last_pose": selected_traj[-1].tolist(),
                    "shape": list(selected_traj.shape)
                }
            
            info_path = self.save_dir / "trajectory_info.json"
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            # Format 4: Human-readable CSV for easy analysis
            if selected_traj is not None:
                csv_path = self.save_dir / "selected_trajectory.csv"
                import pandas as pd
                
                # Only use actuated joint names (match trajectory shape)
                n_joints_in_traj = selected_traj.shape[1]
                if len(joint_names) > n_joints_in_traj:
                    # Filter to only actuated joints (first n_joints_in_traj)
                    actuated_joint_names = joint_names[:n_joints_in_traj]
                else:
                    actuated_joint_names = joint_names
                
                df = pd.DataFrame(selected_traj, columns=actuated_joint_names)
                df.index.name = 'timestep'
                df.to_csv(csv_path)
            
            logger.info(f"Saved {n_traj} trajectories to {self.save_dir}")
            logger.info(f"Files: raw_trajectories.npy, trajectory_info.json")
            if selected_traj is not None:
                logger.info(f"Selected trajectory #{selected_idx}: selected_trajectory.npy, .csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trajectories: {e}")
            return False
    
    def save_trajectories(self, trajectories: jnp.ndarray, kin_tree, 
                         metadata: Optional[Dict] = None) -> bool:
        """
        Save trajectories to multiple formats (.npy, .json, .csv).
        
        Args:
            trajectories: JAX array of shape (n_trajectories, timesteps, n_joints)
            kin_tree: JaxKinTree object with joint information
            metadata: Optional metadata dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy for saving
            traj_np = onp.array(trajectories)
            
            # Get trajectory dimensions
            n_trajectories, timesteps, n_joints_in_traj = traj_np.shape
            
            # Get only the actuated joint names (first n_joints_in_traj joints)
            actuated_joint_names = kin_tree.joint_names[:n_joints_in_traj]
            
            # Save as .npy
            npy_path = self.save_dir / "trajectories.npy"
            onp.save(npy_path, traj_np)
            
            # Create trajectory info
            info = {
                'n_trajectories': int(n_trajectories),
                'timesteps': int(timesteps),
                'n_joints': int(n_joints_in_traj),
                'joint_names': actuated_joint_names,
                'total_joints_in_model': len(kin_tree.joint_names),
                'actuated_joints': kin_tree.num_actuated_joints,
                'timestamp': str(onp.datetime64('now')),
                'metadata': metadata or {}
            }
            
            # Save trajectory info as JSON
            info_path = self.save_dir / "trajectory_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            # Save as CSV (one file per trajectory)
            csv_dir = self.save_dir / "csv"
            csv_dir.mkdir(exist_ok=True)
            
            import pandas as pd
            for traj_idx in range(n_trajectories):
                traj_data = traj_np[traj_idx]  # Shape: (timesteps, n_joints)
                
                # Create DataFrame
                df = pd.DataFrame(traj_data, columns=actuated_joint_names)
                df.insert(0, 'timestep', range(len(df)))
                
                # Save to CSV
                csv_path = csv_dir / f"trajectory_{traj_idx:03d}.csv"
                df.to_csv(csv_path, index=False)
            
            # Save as JSON (more readable format)
            json_path = self.save_dir / "trajectories.json"
            trajectories_list = []
            for traj_idx in range(n_trajectories):
                traj_data = traj_np[traj_idx].tolist()
                trajectories_list.append({
                    'trajectory_id': traj_idx,
                    'joint_values': traj_data,
                    'joint_names': actuated_joint_names
                })
            
            json_data = {
                'info': info,
                'trajectories': trajectories_list
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved {n_trajectories} trajectories to {self.save_dir}")
            logger.info(f"Formats: .npy, .json, .csv ({n_trajectories} files)")
            logger.info(f"Joints: {n_joints_in_traj} actuated joints (of {len(kin_tree.joint_names)} total)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trajectories: {e}")
            return False

    def load_trajectories(self) -> dict:
        """Load previously saved trajectories."""
        try:
            # Try new format first
            info_path = self.save_dir / "trajectory_info.json"
            trajectories_path = self.save_dir / "trajectories.npy"
            
            if info_path.exists() and trajectories_path.exists():
                # Load metadata
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Load trajectories
                all_trajectories = onp.load(trajectories_path)
                
                return {
                    'info': info,
                    'all_trajectories': all_trajectories,
                    'selected_trajectory': None,  # Not used in new format
                    'save_dir': self.save_dir,
                    'format': 'new'
                }
            
            # Fall back to old format
            old_raw_path = self.save_dir / "raw_trajectories.npy"
            if info_path.exists() and old_raw_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                all_trajectories = onp.load(old_raw_path)
                
                # Load selected trajectory if it exists
                selected_path = self.save_dir / "selected_trajectory.npy"
                selected_trajectory = None
                if selected_path.exists():
                    selected_trajectory = onp.load(selected_path)
                
                return {
                    'info': info,
                    'all_trajectories': all_trajectories,
                    'selected_trajectory': selected_trajectory,
                    'save_dir': self.save_dir,
                    'format': 'old'
                }
            
            # No valid format found
            raise FileNotFoundError(f"No trajectory info found at {info_path}")
            
        except Exception as e:
            logger.error(f"Failed to load trajectories: {e}")
            return {}
    
    def has_saved_trajectories(self) -> bool:
        """Check if there are saved trajectories available to load."""
        info_path = self.save_dir / "trajectory_info.json"
        
        # Check new format
        new_path = self.save_dir / "trajectories.npy"
        if info_path.exists() and new_path.exists():
            return True
            
        # Check old format
        old_path = self.save_dir / "raw_trajectories.npy"
        return info_path.exists() and old_path.exists()
    
    def get_trajectory_summary(self) -> dict:
        """Get a summary of saved trajectories without loading the full data."""
        try:
            info_path = self.save_dir / "trajectory_info.json"
            if not info_path.exists():
                return {}
            
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            return {
                'n_trajectories': info.get('n_trajectories', 0),
                'timesteps': info.get('timesteps', 0),
                'timestamp': info.get('timestamp', 'unknown'),
                'has_data': self.has_saved_trajectories()
            }
        except:
            return {}


def create_trajectory_saver(track_dir: Path) -> TrajectorySaver:
    """Create a trajectory saver for the given tracking directory."""
    save_dir = track_dir / "robot_trajectories"
    return TrajectorySaver(save_dir)
