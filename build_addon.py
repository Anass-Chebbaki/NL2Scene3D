import os
import zipfile
import re
import sys

def get_addon_version(init_file_path):
    """Estrae la versione dal dizionario bl_info in __init__.py."""
    if not os.path.exists(init_file_path):
        return None
    with open(init_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Cerca la tupla "version": (x, y, z)
    match = re.search(r'"version"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None

def build_zip():
    addon_dir = "nl2scene3d_addon"
    init_path = os.path.join(addon_dir, "__init__.py")
    
    if not os.path.exists(addon_dir):
        print(f"Errore: La cartella '{addon_dir}' non esiste nella directory corrente.")
        sys.exit(1)
        
    version = get_addon_version(init_path)
    zip_name = f"nl2scene3d_addon_{f'v{version}' if version else 'dist'}.zip"
    
    # Esclusioni comuni durante il packaging
    exclude_patterns = [
        r'__pycache__',
        r'\.pyc$',
        r'\.pyo$',
        r'\.git',
        r'\.DS_Store',
        r'Thumbs\.db'
    ]
    
    print(f"=== Avvio Packaging di NL2Scene3D ===")
    if version:
        print(f"Versione rilevata: {version}")
    print(f"File di output: {zip_name}\n")
    
    files_added = 0
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(addon_dir):
            # Esclude cartelle che corrispondono ai pattern di esclusione
            dirs[:] = [d for d in dirs if not any(re.search(pat, d) for pat in exclude_patterns)]
            
            for file in files:
                # Esclude file che corrispondono ai pattern di esclusione
                if any(re.search(pat, file) for pat in exclude_patterns):
                    continue
                
                full_path = os.path.join(root, file)
                # Il percorso all'interno dello zip deve partire da 'nl2scene3d_addon/...'
                archive_name = os.path.relpath(full_path, os.path.dirname(addon_dir))
                
                zipf.write(full_path, archive_name)
                print(f" -> Aggiunto: {archive_name}")
                files_added += 1
                
    print(f"\nSuccesso! Creato '{zip_name}' contenente {files_added} file.")
    print(f"Dimensione dell'archivio: {os.path.getsize(zip_name) / 1024:.2f} KB")

if __name__ == "__main__":
    build_zip()
