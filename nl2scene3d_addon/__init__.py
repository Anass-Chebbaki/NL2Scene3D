# nl2scene3d/__init__.py
"""
NL2Scene3D - Blender Add-on.

Riordina scene 3D preesistenti con un LLM, in due modalità:

    Step 1 - Randomize:
        Disordina la scena in modo plausibile.

    Step 2 - Riordina con l'LLM, in uno dei due modi:
        - Manuale: l'add-on esporta prompt + JSON, l'utente li usa nel proprio
          LLM allegando a mano i render, poi incolla/carica la risposta.
        - Automatico: l'add-on chiama direttamente il provider (Gemini/
          Anthropic/OpenAI), allega i render e applica la risposta da solo.
      In entrambi i casi i vincoli geometrici (muri, collisioni, Z intatta,
      figli al seguito del padre) sono sempre rispettati.

Nota sull'import-safety:
    Questo file NON importa bpy a livello di modulo e NON importa i sottomoduli
    della UI al top-level. Gli import che dipendono da Blender avvengono solo
    dentro register() / unregister(). In questo modo i moduli puri (nl2scene3d.core.*)
    restano importabili da riga di comando per i test, senza Blender installato.
"""
import sys
import site


bl_info = {
    "name": "NL2Scene3D",
    "author": "NL2Scene3D Team",
    "version": (1, 0, 0),
    # Minimo conservativo: compatibile con Blender 4.2+ e 5.1.x.
    # Alzare se si usano API piu' recenti.
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via an external LLM (manual paste/load or direct API call)",
    "category": "3D View",
}


def register():
    """Registra gli operatori e il pannello UI dell'add-on."""
    # Aggiunge il site-packages utente in sys.path (se non gia' presente) solo al
    # momento della registrazione, in modo da non inquinare il path globale di
    # Blender al solo import del modulo. Altri add-on non vengono influenzati.
    try:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
    except Exception:
        pass

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