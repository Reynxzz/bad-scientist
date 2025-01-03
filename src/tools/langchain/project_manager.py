"""
Project file management module for saving and organizing generated applications.
"""

import json
from pathlib import Path
from typing import Dict

class ProjectManager:
    """Manages project file operations."""
    
    @staticmethod
    def save_project(project_name: str, results: Dict[str, str]) -> Path:
        """
        Save project results to files.
        
        Args:
            project_name: Name of the project
            results: Dictionary containing project results
            
        Returns:
            Path to the project directory
        """
        project_dir = Path(f"projects/{project_name}")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(project_dir / "output.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save implementation
        with open(project_dir / "app.py", "w") as f:
            f.write(results["implementation"])
        
        return project_dir