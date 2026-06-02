# nl2scene3d/__init__.py
"""
NL2Scene3D - Add-on Blender.

Riordina scene 3D preesistenti tramite un MLLM locale (Ollama):
  - Step 1: Randomize    -> disordina la scena in modo plausibile.
  - Step 2: AI Reorder   -> rende le viste, le manda al modello con il JSON
                            della scena, riceve nuove coordinate, le applica
                            con vincoli geometrici (muri, collisioni).

NOTA DI PROGETTO (import-safety):
  Questo file NON importa bpy a livello di modulo e NON importa i sottomoduli
  della UI al top-level. Gli import che dipendono da Blender avvengono solo
  dentro register()/unregister(). Cosi' i moduli puri (nl2scene3d.core.*)
  restano importabili da riga di comando per i test, senza Blender.
"""

bl_info = {
    "name":        "NL2Scene3D",
    "author":      "NL2Scene3D Team",
    "version":     (0, 2, 0),
    # Minimo conservativo: gira su 4.2+ e sul tuo 5.1.x. Alzalo se usi API piu' recenti.
    "blender":     (4, 2, 0),
    "location":    "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via a local multimodal LLM (Ollama)",
    "category":    "3D View",
}


def register():
    from . import operators, ui
    operators.register()
    ui.register()


def unregister():
    from . import operators, ui
    ui.unregister()
    operators.unregister()


if __name__ == "__main__":
    register()
