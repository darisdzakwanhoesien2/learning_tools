import os
import tempfile
from pathlib import Path
import subprocess
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import yaml
import json
from pydantic import BaseModel

class Config(BaseModel):
    execution: dict = {
        "max_runtime": 30,
        "memory_limit": 512,
        "sandbox": True,
        "plot_dir": "data/outputs/plots"
    }

class CodeExecutor:
    def __init__(self, config_path="config.yaml"):
        self.config = Config(config=open(config_path).read())
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [Path("data/scripts"), Path(self.config.execution["plot_dir"])]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def execute_file(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute a Python file from the filesystem."""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the original file to temp location
                temp_file = Path(temp_dir) / os.path.basename(file_path)
                with open(file_path, 'r') as f:
                    content = f.read()
                temp_file.write_text(content)

                # Execute with proper environment
                env = os.environ.copy()

                if self.config.execution["sandbox"]:
                    env["PYTHONUNBUFFERED"] = "1"

                cmd = ["python", str(temp_file)]

                # Set resource limits (Linux only)
                if self.config.execution["sandbox"]:
                    subprocess.run([
                        "ulimit",
                        "-v", str(self.config.execution["memory_limit"] * 1024),
                        "-t", str(self.config.execution["max_runtime"])
                    ], check=True)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=self.config.execution["max_runtime"]
                )

                if result.returncode != 0:
                    return None, f"Execution failed: {result.stderr}"

                # Handle matplotlib output for plots
                if "3D Surface Plot" in result.stdout or "plot_surface" in result.stdout.lower():
                    plot_file = Path(self.config.execution["plot_dir"]) / "output_plot.png"
                    with open(plot_file, 'wb') as f:
                        f.write(result.stdout.encode())
                    return result.stdout, str(plot_file)

                return result.stdout, None

        except Exception as e:
            return None, f"Error executing file: {str(e)}"

    def execute_code(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute inline Python code."""
        try:
            # Create a temporary script
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                return self.execute_file(temp_file)
            finally:
                os.unlink(temp_file)

        except Exception as e:
            return None, str(e)

    def run_page(self, page_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Run a specific Streamlit page file."""
        try:
            # Create a temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the page to our temp location
                temp_file = Path(temp_dir) / os.path.basename(page_path)
                with open(page_path, 'r') as f:
                    content = f.read()
                temp_file.write_text(content)

                # Execute with Streamlit command
                cmd = [
                    "streamlit", "run",
                    "--server.port=0",  # Use random port
                    "--server.address=127.0.0.1",
                    str(temp_file)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.execution["max_runtime"]
                )

                if result.returncode != 0:
                    return None, f"Streamlit execution failed: {result.stderr}"

                # For your specific case with matplotlib plots
                if "3D Surface Plot" in result.stdout or "plot_surface" in result.stdout.lower():
                    plot_file = Path(self.config.execution["plot_dir"]) / "output_plot.png"
                    with open(plot_file, 'wb') as f:
                        f.write(result.stdout.encode())
                    return result.stdout, str(plot_file)

                return result.stdout, None

        except Exception as e:
            return None, str(e)
