"""Record and verify the small source closure for the two Qwen adapters."""
from pathlib import Path

from baselines.common import sha256_file


def implementation_identity(method):
    names = ['common.py', 'adaptation_provenance.py', 'validate_sdft_rledit.py',
             'editor_dev_selection.py', 'wikibigedit_protocol.py']
    if method == 'sdft':
        names += ['run_sdft_qwen3.py', 'sdft_qwen_core.py', 'engram_full_backend.py',
                  'engram_grad_norm.py', 'engram_training_views.py']
    elif method == 'rledit':
        names += ['run_rledit_qwen3.py', 'rledit_qwen_core.py',
                  'run_ultraedit.py', 'run_lora_family.py']
    else:
        raise ValueError('Unknown adaptation')
    directory = Path(__file__).resolve().parent
    return {'directory': str(directory),
            'file_sha256': {name: sha256_file(directory / name) for name in sorted(names)}}


def verify_identity(identity):
    directory = Path(identity['directory'])
    for name, digest in identity['file_sha256'].items():
        if sha256_file(directory / name) != digest:
            raise ValueError('Frozen adapter changed: ' + str(directory / name))
