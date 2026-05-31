# nl2scene3d/__init__.py
"""
NL2Scene3D — Scene Reorganization via Multimodal Language Models.

Pipeline steps:
  1. Load an existing Blender 3D scene.
  2. Artificially disorganize the layout (randomization).
  3. Reorganize it via a Gemini MLLM call.
  4. Optionally refine the result with visual feedback.
"""

__version__ = "0.1.0"
__author__  = "NL2Scene3D Team"