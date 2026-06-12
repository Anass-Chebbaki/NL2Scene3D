# nl2scene3d/__init__.py
"""
NL2Scene3D - Blender Add-on.

Riordina scene 3D preesistenti con un LLM, in modalita' manuale (human-in-the-loop):

    Step 1 - Randomize:
        Disordina la scena in modo plausibile.

    Step 2 - Esporta / Applica:
        L'add-on esporta prompt + JSON della scena. L'utente li usa nel proprio
        LLM (es. Gemini in AI Studio), allegando a mano i render; poi incolla o
        carica la risposta JSON e l'add-on la applica rispettando i vincoli
        geometrici (muri, collisioni, Z intatta, figli al seguito del padre).

Nota sull'import-safety:
    Questo file NON importa bpy a livello di modulo e NON importa i sottomoduli
    della UI al top-level. Gli import che dipendono da Blender avvengono solo
    dentro register() / unregister(). In questo modo i moduli puri (nl2scene3d.core.*)
    restano importabili da riga di comando per i test, senza Blender installato.
"""
import sys
import site

# Assicura che la directory site-packages dell'utente sia inclusa in sys.path.
# Questo permette a Blender di importare moduli installati via `--user` (come Pillow).
try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass


bl_info = {
    "name": "NL2Scene3D",
    "author": "NL2Scene3D Team",
    "version": (1, 0, 0),
    # Minimo conservativo: compatibile con Blender 4.2+ e 5.1.x.
    # Alzare se si usano API piu' recenti.
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via an external LLM (manual paste/load workflow)",
    "category": "3D View",
}


def register():
    """Registra gli operatori e il pannello UI dell'add-on."""
    from . import operators, ui
    operators.register()
    ui.register()


def unregister():
    """Deregistra il pannello UI e gli operatori dell'add-on."""
    from . import operators, ui
    ui.unregister()
    operators.unregister()


if __name__ == "__main__":
    register()