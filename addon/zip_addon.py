from pathlib import Path
import shutil

# =========================================
# CONFIG
# =========================================

FOLDER_NAME = "nl2scene3d_addon"
ZIP_NAME = "nl2scene3d_addon"

# =========================================
# PATHS
# =========================================

current_dir = Path(__file__).parent

source_folder = current_dir / FOLDER_NAME
output_zip = current_dir / ZIP_NAME

# =========================================
# CREATE ZIP
# =========================================

shutil.make_archive(
    base_name=str(output_zip),
    format="zip",
    root_dir=str(current_dir),
    base_dir=FOLDER_NAME
)

print(f"\nZIP creata: {ZIP_NAME}.zip")