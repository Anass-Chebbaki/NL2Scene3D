import sys
import shutil
import subprocess
from pathlib import Path

# =============================================================================
# NL2Scene3D Add-on Packaging and Dependency Bundling Script
# =============================================================================

FOLDER_NAME = "nl2scene3d_addon"
ZIP_NAME = "nl2scene3d_addon"

current_dir = Path(__file__).parent
addon_dir   = current_dir / FOLDER_NAME
vendor_dir  = addon_dir / "vendor"
output_zip  = current_dir / ZIP_NAME
req_file    = current_dir.parent / "requirements.txt"

def get_blender_python_version() -> str:
    """
    Attempts to dynamically query the Blender executable in the system PATH
    to discover its exact Python ABI version.
    Falls back to "3.13" (the standard version for Blender 5.x) if query fails.
    """
    try:
        # Run blender in background and output its python version
        result = subprocess.run(
            ["blender", "--background", "--python-expr", "import sys; print(f'SYS_VER:{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        for line in result.stdout.splitlines():
            if "SYS_VER:" in line:
                version = line.split("SYS_VER:")[1].strip()
                print(f"[Info] Dynamically detected Blender Python version: {version}")
                return version
    except Exception:
        pass
    print("[Info] Blender not found in PATH or failed to query. Defaulting to Python 3.13 (Blender 5.x standard).")
    return "3.13"

def parse_addon_requirements() -> list[str]:
    """
    Parses core requirements from requirements.txt, excluding standalone GUI packages.
    """
    requirements = []
    if not req_file.exists():
        print(f"[Warning] requirements.txt not found at {req_file}. Using fallback default requirements.")
        return [
            "google-genai>=1.0.0",
            "google-api-core>=2.15.0",
            "python-dotenv>=1.0.0",
            "pydantic>=2.0.0",
            "pydantic-core>=2.46.4",
            "cryptography>=41.0.0",
            "cffi>=1.15.0",
            "requests>=2.31.0",
            "Pillow>=10.0.0",
            "packaging>=23.0"
        ]

    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Stop parsing when we reach the standalone GUI application section
            if line.startswith("#") and "GUI" in line:
                break
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
    return requirements

def check_vendor_health(target_py_version: str) -> bool:
    """
    Verifies if the vendor directory exists, is complete, and contains the binary extension (.pyd)
    compiled for the exact Python version that Blender is using.
    Returns True if healthy, False if missing, incomplete, or targeting a different Python version.
    """
    if not vendor_dir.exists() or not vendor_dir.is_dir():
        print("[Info] Vendor directory does not exist.")
        return False

    # Check for crucial marker files that prove a complete and uncorrupted installation
    crucial_markers = [
        vendor_dir / "google" / "genai" / "types.py",
        vendor_dir / "requests" / "__init__.py",
        vendor_dir / "pydantic" / "__init__.py",
        vendor_dir / "cryptography" / "__init__.py"
    ]

    for marker in crucial_markers:
        if not marker.exists():
            print(f"[Warning] Crucial dependency file is missing: {marker.relative_to(addon_dir)}")
            return False

    # Check for the correct platform binary (.pyd) compiled for this specific Python version
    py_version_nodot = target_py_version.replace(".", "")
    binary_marker = vendor_dir / "pydantic_core" / f"_pydantic_core.cp{py_version_nodot}-win_amd64.pyd"
    
    if not binary_marker.exists():
        print(f"[Warning] Mismatch detected: missing binary for Python {target_py_version} ({binary_marker.name})")
        return False

    return True

def restore_vendor_dependencies(target_py_version: str):
    """
    Deletes any outdated/corrupted vendor folder and performs a clean pip target install
    requesting wheels matching the exact Python version and platform of Blender.
    """
    if vendor_dir.exists():
        print("[Info] Deleting outdated/corrupted vendor folder...")
        try:
            shutil.rmtree(vendor_dir)
        except Exception as exc:
            print(f"[Error] Failed to remove vendor directory: {exc}")
            sys.exit(1)

    print(f"[Info] Re-populating vendor directory with clean dependencies for Python {target_py_version}...")
    requirements = parse_addon_requirements()
    
    # Run pip targeting the vendor folder, forcing download of cp313 win_amd64 wheels
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "--force-reinstall",
        "--target", str(vendor_dir),
        "--platform", "win_amd64",
        "--python-version", target_py_version,
        "--implementation", "cp",
        "--only-binary=:all:"
    ] + requirements

    print(f"[Running] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Success] Dependencies successfully vendored.")
    except subprocess.CalledProcessError as exc:
        print("[Error] Dependency installation failed!")
        print(exc.stdout)
        print(exc.stderr)
        sys.exit(1)

def main():
    print("=" * 80)
    print("NL2Scene3D - Add-on Packager (ABI-Aware)")
    print("=" * 80)

    # Step 1: Detect target Python version used by Blender
    target_py_version = get_blender_python_version()

    # Step 2: Analyze and repair the vendor folder if necessary
    if not check_vendor_health(target_py_version):
        print(f"[Info] Vendor folder is missing, incomplete, or outdated for Python {target_py_version}. Initiating repair...")
        restore_vendor_dependencies(target_py_version)
    else:
        print(f"[Success] Vendor directory is complete and targeting Python {target_py_version}. Skipping dependency restoration.")

    # Step 3: Create the distribution ZIP file
    print(f"[Info] Packaging '{FOLDER_NAME}' into '{ZIP_NAME}.zip'...")
    try:
        # Remove any existing zip to avoid file conflicts
        zip_file = current_dir / f"{ZIP_NAME}.zip"
        if zip_file.exists():
            zip_file.unlink()

        shutil.make_archive(
            base_name=str(output_zip),
            format="zip",
            root_dir=str(current_dir),
            base_dir=FOLDER_NAME
        )
        print(f"\n[Success] ZIP successfully created: {ZIP_NAME}.zip")
    except Exception as exc:
        print(f"[Error] Packaging failed: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()