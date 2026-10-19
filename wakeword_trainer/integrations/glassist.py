"""Put a finished model into a GLaSSIST installation (a source checkout, not the packaged app).

GLaSSIST loads every ``models/<name>.onnx`` and activates the names listed in
``HA_WAKE_WORD_MODELS`` in its ``.env`` file. This copies the model there and registers the name.

Behaviour that matters (all of it was a bug in an earlier version):
  * an existing model with the same name is NEVER overwritten silently (``force=True`` to replace);
  * when ``HA_WAKE_WORD_MODELS`` is not set, GLaSSIST's default (``alexa``) is kept in the list;
  * ``replace=True`` is the only way the list is reduced to just the new model.
"""
import filecmp
import os
import re

DEFAULT_MODELS = ['alexa']          # what GLaSSIST uses when HA_WAKE_WORD_MODELS is missing


class ModelExistsError(FileExistsError):
    """A different model with this name is already installed (pass force=True to replace it)."""


def _read_env(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def _find(lines, key):
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=(.*)$')
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            return i, m.group(1).strip()
    return None, None


def _set(lines, key, value):
    i, _ = _find(lines, key)
    if i is None:
        lines.append(f'{key}={value}')
    else:
        lines[i] = f'{key}={value}'


def check_dir(glassist_dir):
    """Return None if ``glassist_dir`` looks like a GLaSSIST checkout, else a message."""
    if not glassist_dir or not os.path.isdir(glassist_dir):
        return 'The folder does not exist.'
    if not os.path.isfile(os.path.join(glassist_dir, 'wake_word_detector.py')):
        return 'This does not look like a GLaSSIST folder (wake_word_detector.py not found).'
    return None


def list_models(glassist_dir):
    """The wake words GLaSSIST would activate: the .env value, or its default when unset."""
    lines = _read_env(os.path.join(glassist_dir, '.env'))
    _, value = _find(lines, 'HA_WAKE_WORD_MODELS')
    if value is None:
        return list(DEFAULT_MODELS)
    return [m.strip() for m in value.split(',') if m.strip()]


def install_model(model_path, glassist_dir, name, replace=False, force=False, enable=True):
    """Copy ``model_path`` to ``<glassist_dir>/models/<name>.onnx`` and register ``name``.

    Returns {'file', 'models', 'overwrote'}. Raises ModelExistsError instead of overwriting a
    different model unless force=True.
    """
    problem = check_dir(glassist_dir)
    if problem:
        raise ValueError(problem)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    models_dir = os.path.join(glassist_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, f'{name}.onnx')
    overwrote = False
    if os.path.exists(dest) and not filecmp.cmp(model_path, dest, shallow=False):
        if not force:
            raise ModelExistsError(f'{dest} already exists and is different')
        overwrote = True
    if not os.path.exists(dest) or overwrote:
        with open(model_path, 'rb') as src, open(dest, 'wb') as out:
            out.write(src.read())

    env_path = os.path.join(glassist_dir, '.env')
    lines = _read_env(env_path)
    models = [] if replace else list_models(glassist_dir)
    if name not in models:
        models.append(name)
    _set(lines, 'HA_WAKE_WORD_MODELS', ','.join(models))
    if enable:
        _set(lines, 'HA_WAKE_WORD_ENABLED', 'true')
    tmp = env_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, env_path)
    return {'file': dest, 'models': models, 'overwrote': overwrote}
