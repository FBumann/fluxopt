"""Generate the API reference pages automatically from source.

Only the modules backing the public surface get a page. Documenting every
module would make the reference read far wider than the API actually is —
see ``docs/api/index.md`` for the surface the project commits to.
"""

from pathlib import Path

import mkdocs_gen_files

INTERNAL_MODULES = {
    'fluxopt.benchmark',
    'fluxopt.constraints',
    'fluxopt.contract',
    'fluxopt.contributions',
    'fluxopt.effect_terms',
    'fluxopt.validation',
}


def is_internal(parts: tuple[str, ...]) -> bool:
    """Whether a module, or any package containing it, is implementation detail."""
    return any('.'.join(parts[: depth + 1]) in INTERNAL_MODULES for depth in range(len(parts)))


nav = mkdocs_gen_files.Nav()
nav['Overview'] = 'index.md'
root = Path('src')

for path in sorted(root.rglob('*.py')):
    module_path = path.relative_to(root).with_suffix('')
    doc_path = path.relative_to(root).with_suffix('.md')
    full_doc_path = Path('api', doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == '__init__':
        parts = parts[:-1]
        doc_path = doc_path.with_name('index.md')
        full_doc_path = full_doc_path.with_name('index.md')
    elif parts[-1].startswith('_'):
        continue

    if not parts or is_internal(parts):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        identifier = '.'.join(parts)
        fd.write(f'::: {identifier}\n')

    mkdocs_gen_files.set_edit_path(full_doc_path, Path('..', path))

with mkdocs_gen_files.open('api/SUMMARY.md', 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())
