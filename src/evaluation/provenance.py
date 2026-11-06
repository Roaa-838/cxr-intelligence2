"""Record versions and effective runtime settings, never environment secrets."""
import importlib.metadata
import platform
import subprocess
import sys


def runtime_metadata(pipeline):
    versions = {}
    for name in ('numpy', 'pandas', 'pillow', 'torch', 'transformers', 'nltk',
                 'rouge-score', 'bert-score', 'radgraph', 'colpali-engine', 'open-clip-torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    components = {}
    for key in ('_vlm', '_generator', '_retriever'):
        obj = getattr(pipeline, key, None)
        if obj is None:
            continue
        model = getattr(obj, '_model', None)
        components[key.lstrip('_')] = {
            'class': type(obj).__name__, 'model_id': getattr(obj, 'model_id', None),
            'model_name': getattr(obj, '_model_name', None),
            'device': str(getattr(obj, 'device', None)),
            'dtype': str(getattr(model, 'dtype', None)),
            'loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
            'loaded_in_8bit': bool(getattr(model, 'is_loaded_in_8bit', False)),
            'device_map': {str(k): str(v) for k, v in (getattr(model, 'hf_device_map', {}) or {}).items()},
        }
    return {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'source_commit': commit, 'working_tree_dirty': dirty, 'components': components}
