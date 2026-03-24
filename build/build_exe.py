"""Build script — packages the agent into a single .exe using PyInstaller.

Usage:
    cd training-network
    pip install pyinstaller
    python build/build_exe.py

Output: dist/GPUNetworkAgent/GPUNetworkAgent.exe
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
MAIN_SCRIPT = os.path.join(AGENT_DIR, "main.py")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", "GPUNetworkAgent",
    "--onedir",
    "--console",
    # Include all agent modules
    "--add-data", f"{os.path.join(AGENT_DIR, 'config.json')};.",
    # Hidden imports that PyInstaller may miss
    "--hidden-import", "pynvml",
    "--hidden-import", "psutil",
    "--hidden-import", "websockets",
    "--hidden-import", "llama_cpp",
    "--hidden-import", "huggingface_hub",
    # Paths
    "--paths", AGENT_DIR,
    "--distpath", os.path.join(ROOT, "dist"),
    "--workpath", os.path.join(ROOT, "build", "_pyinstaller"),
    "--specpath", os.path.join(ROOT, "build"),
    # Main script
    MAIN_SCRIPT,
]

print(f"Building agent exe...")
print(f"Command: {' '.join(cmd)}")
subprocess.run(cmd, check=True)
print(f"\nDone! Output: {os.path.join(ROOT, 'dist', 'GPUNetworkAgent', 'GPUNetworkAgent.exe')}")
