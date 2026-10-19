"""Train a custom openWakeWord-compatible wake word, in any supported language.

A standalone tool: it outputs a plain ``.onnx`` model that GLaSSIST (or anything that loads
openWakeWord models) can use. The pipeline:

  1. generate  - synthetic positives/negatives with neural TTS voices of the chosen language
  2. record    - your own voice (the word, normal speech, the quiet room) and everyday audio
  3. train     - a tiny dense head on openWakeWord audio embeddings, exported to ONNX
  4. evaluate  - clip-level false activations on audio that was never used for training

Only the pure-Python modules are imported here, so importing the package is cheap and does not
need NumPy or PyTorch.
"""
from . import confusables, languages, workspace  # noqa: F401

__version__ = '0.1.0'

SAMPLE_RATE = 16000
FRAME = 1280           # 80 ms, the openWakeWord frame size
WINDOW = 16            # the model looks at 16 embeddings (~1.3 s)
EMBED_DIM = 96
