"""openWakeWord's three base audio models: download with its helper, then VERIFY their SHA-256.

openWakeWord downloads its models from GitHub releases without checking them. A tampered
melspectrogram/embedding model would silently poison everything trained on top of it, so the
hashes of the files shipped with openWakeWord 0.6.0 are pinned here (``base_models.json``).
"""
import hashlib
import json
import os

PINNED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base_models.json')


def expected():
    with open(PINNED, encoding='utf-8') as f:
        return json.load(f)['files']


def models_dir():
    import openwakeword
    return os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory=None):
    """Return a list of problems (empty = all three base models are present and genuine)."""
    directory = directory or models_dir()
    problems = []
    for name, digest in expected().items():
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            problems.append(f'{name}: missing')
        elif sha256(path) != digest:
            problems.append(f'{name}: SHA-256 does not match the pinned value')
    return problems


def ensure(download=True, log=print):
    """Make sure the base models are there and genuine; download them first if needed."""
    problems = verify()
    if problems and download:
        log('Downloading the openWakeWord base models...')
        import openwakeword
        openwakeword.utils.download_models()
        problems = verify()
    if problems:
        raise RuntimeError('openWakeWord base models are not usable: ' + '; '.join(problems) +
                           '. Delete the files in ' + models_dir() + ' and run `wakeword-trainer models` again.')
    return True
