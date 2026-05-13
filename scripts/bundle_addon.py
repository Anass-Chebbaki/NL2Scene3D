# scripts/bundle_addon.py
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path

def bundle():
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src" / "nl2scene3d"
    addon_dir = root / "addon" / "nl2scene3d_addon"
    target_core_dir = addon_dir / "nl2scene3d"
    vendor_dir = addon_dir / "vendor"
    
    # Per Blender 5.1, forziamo il download delle dipendenze per Python 3.13
    # Questo evita l'errore "No module named 'pydantic_core._pydantic_core'"
    BLENDER_PY_VERSION = "3.13" 
    
    print(f"Bundling addon in: {addon_dir}")
    
    # 1. Cleanup
    if target_core_dir.exists():
        shutil.rmtree(target_core_dir)
    if vendor_dir.exists():
        try:
            shutil.rmtree(vendor_dir)
        except Exception as e:
            print(f"Warning: Could not remove vendor dir: {e}")
    
    # 2. Copy core logic
    print(f"Copying {src_dir} -> {target_core_dir}")
    shutil.copytree(src_dir, target_core_dir)
    
    # 3. Vendor dependencies (Cross-platform/version safe)
    print(f"Vendoring dependencies for Python {BLENDER_PY_VERSION}...")
    requirements_file = root / "requirements.txt"
    
    if requirements_file.exists():
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            with open(requirements_file, 'r') as f:
                for line in f:
                    # Escludiamo librerie UI e dev
                    if any(x in line for x in ["PySide6", "customtkinter", "black", "ruff", "pytest"]):
                        continue
                    tmp.write(line)
            tmp_path = Path(tmp.name)
        
        try:
            # Usiamo --only-binary=:all: per assicurarci di scaricare i .pyd già pronti
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-t", str(vendor_dir), 
                "-r", str(tmp_path),
                "--only-binary=:all:",
                "--platform", "win_amd64",
                "--python-version", BLENDER_PY_VERSION,
                "--only-binary=:all:",
                "--upgrade"
            ])
        except Exception as e:
            print(f"Pip error: {e}")
            # Fallback se la versione 3.13 non è ancora disponibile o fallisce la piattaforma
            print("Falling back to standard install...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-t", str(vendor_dir), 
                "-r", str(tmp_path)
            ])
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)

    # 4. Create ZIP
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)
    zip_name = dist_dir / "nl2scene3d_addon"
    print(f"Creating ZIP: {zip_name}.zip")
    shutil.make_archive(str(zip_name), 'zip', root / "addon", "nl2scene3d_addon")
    
    print("Done!")

if __name__ == "__main__":
    bundle()
