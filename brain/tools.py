"""Ollama-callable vault tools (module-level functions for schema introspection)."""

# Note: do not use ``from __future__ import annotations`` here. The Ollama client
# builds Pydantic tool schemas from signatures; postponed annotations break
# ``dict[str, Any]`` on ``update_frontmatter`` (PydanticUserError: class not fully defined).

import base64
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional

from brain.context import get_context
from brain.messages import msg_to_dict

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Structured return value for every tool function."""

    success: bool
    message: str

    def __str__(self) -> str:
        return self.message

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vault helpers
# ---------------------------------------------------------------------------

def _vault() -> Path:
    return get_context().vault_path.resolve()


def _encoding() -> str:
    return get_context().user.note_encoding or 'utf-8'


def _read_vault_text(path: Path) -> str:
    """Read a vault file, replacing un-decodable bytes with \\ufffd.

    ``errors='replace'`` stops invalid bytes from becoming lone surrogates
    that crash JSON serialisation later (see ``_sanitize_messages``).
    """
    return path.read_text(encoding=_encoding(), errors='replace')


def _image_suffixes() -> frozenset[str]:
    return frozenset(x.lower() for x in get_context().defaults.files.image_extensions)


def _ensure_md(title: str) -> str:
    return title if title.endswith('.md') else title + '.md'


def _is_hidden(path: Path) -> bool:
    """Return True if any component of *path* begins with a dot."""
    return any(part.startswith('.') for part in path.parts)


def _vault_notes() -> list[Path]:
    """Sorted list of all non-hidden ``.md`` notes inside the vault."""
    vault = _vault()
    return sorted(p for p in vault.rglob('*.md') if not _is_hidden(p.relative_to(vault)))


# ---------------------------------------------------------------------------
# Vault index: O(1) note lookup by filename (case-insensitive)
# ---------------------------------------------------------------------------

class _VaultIndex:
    """
    Per-process in-memory cache mapping ``lower_filename → [Path]``.

    Built lazily on first use and explicitly invalidated by every write
    operation so that subsequent lookups reflect the current vault state.
    A lock protects rebuild / read for future concurrent use.
    """

    def __init__(self) -> None:
        self._index: dict[str, list[Path]] = {}
        self._valid: bool = False
        self._lock = threading.Lock()

    def _rebuild_unlocked(self) -> None:
        self._index = {}
        for p in _vault_notes():
            self._index.setdefault(p.name.lower(), []).append(p)
        self._valid = True

    def find(self, filename_lower: str) -> Optional[Path]:
        """Return the first path matching *filename_lower*, or ``None``."""
        with self._lock:
            if not self._valid:
                self._rebuild_unlocked()
            paths = self._index.get(filename_lower, [])
            return paths[0] if paths else None

    def count_markdown_under(self, folder: Path) -> int:
        """Count indexed ``.md`` notes under *folder* (vault-relative, non-hidden)."""
        with self._lock:
            if not self._valid:
                self._rebuild_unlocked()
            vault = _vault()
            n = 0
            for paths in self._index.values():
                for p in paths:
                    try:
                        p.relative_to(folder)
                    except ValueError:
                        continue
                    try:
                        rel = p.relative_to(vault)
                    except ValueError:
                        continue
                    if not _is_hidden(rel):
                        n += 1
            return n

    def invalidate(self) -> None:
        """Mark stale; next read will trigger a full rebuild."""
        with self._lock:
            self._valid = False


_NOTE_INDEX = _VaultIndex()


# ---------------------------------------------------------------------------
# Note path resolution
# ---------------------------------------------------------------------------

def _find_note_path(title: str) -> Optional[Path]:
    """Locate a note by bare filename (case-insensitive) using the vault index."""
    return _NOTE_INDEX.find(_ensure_md(title).lower())


def _resolve_note_path(note_ref: str) -> Optional[Path]:
    """
    Resolve a note reference to an absolute ``Path``.

    Accepts either:
    - a vault-relative path (contains ``/`` or ends with ``.md``)
    - a bare note title (resolved via the in-memory vault index)
    """
    vault = _vault()
    raw = note_ref.strip().replace('\\', '/').lstrip('/')
    if not raw:
        return None
    path_shaped = ('/' in raw) or raw.lower().endswith('.md')
    if path_shaped:
        rel = raw if raw.lower().endswith('.md') else f'{raw}.md'
        candidate = (vault / rel).resolve()
        try:
            _assert_in_vault(candidate)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None
    return _find_note_path(raw)


def _assert_in_vault(path: Path) -> None:
    try:
        path.resolve().relative_to(_vault())
    except ValueError as exc:
        raise ValueError('Access denied: path escapes the vault boundary.') from exc


# ---------------------------------------------------------------------------
# Frontmatter + wikilink helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    if not YAML_AVAILABLE or not content.startswith('---'):
        return {}, content
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}, content
    try:
        meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, content[match.end():]


def _serialize_frontmatter(meta: dict[str, Any], body: str) -> str:
    if not meta:
        return body
    fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True).strip()
    return f'---\n{fm}\n---\n\n{body}'


def _extract_wikilinks(content: str) -> list[str]:
    return list(dict.fromkeys(
        re.findall(r'\[\[([^\]|#\n]+?)(?:\|[^\]]+)?\]\]', content),
    ))


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def active_vision_model() -> str:
    """
    Resolve the active vision model with the following precedence:
    ``OLLAMA_VISION_MODEL`` env → ``--vision-model`` CLI override →
    ``ollama_vision_model`` user JSON → main chat model.
    """
    ctx = get_context()
    candidates = (
        os.environ.get('OLLAMA_VISION_MODEL', '').strip(),
        ctx.vision_model_override.strip(),
        (ctx.user.ollama_vision_model or '').strip(),
        (ctx.user.ollama_model or '').strip(),
    )
    for candidate in candidates:
        if candidate:
            return candidate
    return ''


_VISION_MODE_NAMES: Final[tuple[str, ...]] = ('ocr', 'describe', 'full')


def _vision_prompt(mode: str) -> Optional[str]:
    """Return the vision prompt for *mode*, or ``None`` if the mode is unknown."""
    v = get_context().defaults.vision
    prompts: dict[str, str] = {
        'ocr': v.ocr,
        'describe': v.describe,
        'full': v.full,
    }
    return prompts.get(mode)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def list_directory(relative_path: str = '') -> ToolResult:
    """
    Lists the files and sub-folders inside a vault directory.

    Pass an empty string or omit to list the vault root.
    Folder entries include a note count, which helps decide where to place a new note.
    Use this to orient when the vault layout is unknown; prefer read_note or
    search_notes when you already have a path or keyword.
    """
    vault = _vault()
    lim = get_context().defaults.limits
    target = (vault / relative_path).resolve()
    try:
        _assert_in_vault(target)
    except ValueError as exc:
        return ToolResult(False, f'{exc}')
    if not target.exists():
        return ToolResult(False, f"Directory '{relative_path}' does not exist.")
    if not target.is_dir():
        return ToolResult(False, f"'{relative_path}' is a file. Use read_note instead.")

    folders: list[str] = []
    notes: list[str] = []
    images: list[str] = []
    others: list[str] = []
    for item in sorted(target.iterdir()):
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            note_count = _NOTE_INDEX.count_markdown_under(item)
            folders.append(f'📁 {item.name}/  ({note_count} note(s))')
        elif item.suffix == '.md':
            notes.append(f'📄 {item.name}')
        elif item.suffix.lower() in _image_suffixes():
            images.append(f'🖼 {item.name}')
        else:
            others.append(item.name)

    label = relative_path or '(Vault Root)'
    lines = [f'Contents of /{label}:']
    lines.extend(folders)
    lines.extend(notes)
    if images:
        lines.append('Images (vault-relative path: folder above + filename → read_image):')
        lines.extend(images)
    if others:
        cap = lim.max_list_other_files
        others.sort()
        rest = others[cap:]
        for name in others[:cap]:
            lines.append(f'📎 {name}')
        if rest:
            lines.append(
                f'  … and {len(rest)} more non-note, non-image file(s) not listed.',
            )
    if not folders and not notes and not images and not others:
        lines.append('  Directory is empty.')
    return ToolResult(True, '\n'.join(lines))


def read_note(note_title: str) -> ToolResult:
    """
    Reads a note by bare title or vault-relative path (as shown by list_directory).

    The result appends an outgoing-links list at the end so you can see which
    other notes this one references without parsing the Markdown yourself.
    """
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(
            False,
            f"Note '{note_title}' not found. Use list_directory, then pass "
            'either the file name or the full vault-relative path (e.g. Subject/Topic/Name).',
        )
    try:
        content = _read_vault_text(path)
        links = _extract_wikilinks(content)
        suffix = f'\n\n---\n[Outgoing links: {", ".join(links)}]' if links else ''
        return ToolResult(True, content + suffix)
    except OSError as exc:
        return ToolResult(False, f'Could not read note. {exc}')


def create_note(note_title: str, content: str, folder: str = '') -> ToolResult:
    """
    Creates a new markdown note (optionally under ``folder``).

    If *content* includes Markdown links whose target is a vault path, encode
    spaces and reserved characters in the URL portion so editors do not truncate
    the link at the first space; filesystem paths are unchanged.
    """
    filename = _ensure_md(note_title)
    vault = _vault()
    dest_dir = vault / folder if folder else vault
    path = dest_dir / filename
    try:
        _assert_in_vault(path)
    except ValueError as exc:
        return ToolResult(False, f'{exc}')
    if path.exists():
        return ToolResult(False, f"'{note_title}' already exists. Use edit_note to modify it.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=_encoding())
        _NOTE_INDEX.invalidate()
        return ToolResult(True, f"Created '{path.relative_to(vault)}'.")
    except OSError as exc:
        return ToolResult(False, f'Could not create note. {exc}')


def edit_note(note_title: str, content: str, mode: str = 'append') -> ToolResult:
    """
    Edits an existing note.

    ``mode`` options:
    - ``append``    — add *content* after the existing body (default; safest).
    - ``prepend``   — insert *content* before the body, after any YAML frontmatter.
    - ``overwrite`` — replace the entire file; use only when a full rewrite is needed.

    Same Markdown link rules as ``create_note``: percent-encode spaces in inline
    link targets when pointing at paths that contain them.
    """
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(False, f"Note '{note_title}' not found.")
    enc = _encoding()
    try:
        if mode == 'overwrite':
            path.write_text(content, encoding=enc)
        elif mode == 'append':
            existing = _read_vault_text(path)
            path.write_text(existing.rstrip() + '\n\n' + content, encoding=enc)
        elif mode == 'prepend':
            existing = _read_vault_text(path)
            meta, body = _parse_frontmatter(existing)
            new_body = content + '\n\n' + body
            path.write_text(
                _serialize_frontmatter(meta, new_body) if meta else new_body,
                encoding=enc,
            )
        else:
            return ToolResult(
                False,
                f"Unknown mode '{mode}'. Valid options: 'append', 'prepend', 'overwrite'.",
            )
        _NOTE_INDEX.invalidate()
        return ToolResult(True, f"Note '{note_title}' updated ({mode} mode).")
    except OSError as exc:
        return ToolResult(False, f'Could not edit note. {exc}')


def move_note(note_title: str, destination_folder: str) -> ToolResult:
    """Moves a note to another folder inside the vault."""
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(False, f"Note '{note_title}' not found.")
    vault = _vault()
    dest_dir = vault / destination_folder
    dest_path = dest_dir / path.name
    try:
        _assert_in_vault(dest_path)
    except ValueError as exc:
        return ToolResult(False, f'{exc}')
    if dest_path.exists():
        return ToolResult(
            False,
            f"'{path.name}' already exists in '{destination_folder}'. "
            'Rename or remove it first.',
        )
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.rename(dest_path)
        _NOTE_INDEX.invalidate()
        return ToolResult(True, f"Moved '{note_title}' → '{dest_path.relative_to(vault)}'.")
    except OSError as exc:
        return ToolResult(False, f'Could not move note. {exc}')


def delete_note(note_title: str, confirm: bool = False) -> ToolResult:
    """
    Deletes a note permanently.

    ``confirm`` **must** be ``True``; the agent must obtain explicit user
    confirmation before passing this flag.  Calling with the default
    ``confirm=False`` is intentionally a no-op error so that accidental or
    un-confirmed deletions are impossible.
    """
    if not confirm:
        return ToolResult(
            False,
            f"Deletion of '{note_title}' requires explicit user "
            'confirmation. Ask the user to confirm, then call delete_note with '
            'confirm=true.',
        )
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(False, f"Note '{note_title}' not found.")
    try:
        path.unlink()
        _NOTE_INDEX.invalidate()
        return ToolResult(True, f"Deleted '{note_title}'.")
    except OSError as exc:
        return ToolResult(False, f'Could not delete note. {exc}')


def search_notes(query: str, case_sensitive: bool = False) -> ToolResult:
    """
    Full-text search across all markdown notes, including filename stems.

    Returns up to ``max_search_results`` matches with a short snippet per line.
    Prefer this over manually browsing with list_directory when looking for
    notes by topic, keyword, or title fragment.
    """
    ctx = get_context()
    vault = _vault()
    lim = ctx.defaults.limits
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    results: list[str] = []
    cap = lim.max_search_results
    snip = lim.search_snippet_chars
    per_file_cap = lim.max_matches_per_file
    for note_path in _vault_notes():
        rel = note_path.relative_to(vault)
        if pattern.search(note_path.stem):
            results.append(f'  [{rel}]  ← filename match')
            if len(results) >= cap:
                break
            continue
        try:
            lines = _read_vault_text(note_path).splitlines()
        except OSError:
            continue
        file_hits = 0
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                results.append(f'  [{rel}] line {i}: {line.strip()[:snip]}')
                file_hits += 1
                if len(results) >= cap:
                    results.append(f'  … truncated at {cap} results.')
                    return ToolResult(
                        True, f"Search for '{query}':\n" + '\n'.join(results),
                    )
                if file_hits >= per_file_cap:
                    break
    if not results:
        return ToolResult(True, f"No notes found containing '{query}'.")
    return ToolResult(
        True, f"Search for '{query}' ({len(results)} match(es)):\n" + '\n'.join(results),
    )


def get_backlinks(note_title: str) -> ToolResult:
    """
    Find all notes that [[wikilink]] to the given note title.

    Reads every file in the vault — use only when you specifically need the
    backlink graph; prefer search_notes for general keyword lookups.
    """
    vault = _vault()
    target = note_title.removesuffix('.md')
    pat = re.compile(
        r'\[\[' + re.escape(target) + r'(?:\|[^\]]+)?\]\]',
        re.IGNORECASE,
    )
    backlinks: list[str] = []
    for note_path in _vault_notes():
        try:
            content = _read_vault_text(note_path)
        except OSError:
            continue
        if pat.search(content):
            backlinks.append(f'  📄 {note_path.relative_to(vault)}')
    if not backlinks:
        return ToolResult(True, f"No backlinks found for '[[{target}]]'.")
    return ToolResult(
        True,
        f"Notes linking to '[[{target}]]' ({len(backlinks)}):\n" + '\n'.join(backlinks),
    )


def update_frontmatter(note_title: str, metadata: dict[str, Any]) -> ToolResult:
    """
    Merge keys into a note's YAML frontmatter, creating it if absent.

    Common keys: ``tags`` (list of strings), ``date``, ``aliases`` (list),
    ``title``. Existing keys not in *metadata* are preserved.
    """
    if not YAML_AVAILABLE:
        return ToolResult(False, 'PyYAML is not installed. Run: pip install PyYAML')
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(False, f"Note '{note_title}' not found.")
    enc = _encoding()
    try:
        content = _read_vault_text(path)
        existing_meta, body = _parse_frontmatter(content)
        existing_meta.update(metadata)
        path.write_text(_serialize_frontmatter(existing_meta, body), encoding=enc)
        _NOTE_INDEX.invalidate()
        return ToolResult(True, f"Frontmatter updated on '{note_title}'.")
    except OSError as exc:
        return ToolResult(False, f'Could not update frontmatter. {exc}')


def read_image(relative_path: str, mode: str = 'full') -> ToolResult:
    """OCR or describe an image inside the vault via the vision model."""
    ctx = get_context()
    vault = _vault()
    lim = ctx.defaults.limits
    raw = relative_path.strip().replace('\\', '/').lstrip('/')
    path = (vault / raw).resolve()
    try:
        _assert_in_vault(path)
    except ValueError as exc:
        return ToolResult(False, f'{exc}')
    if not path.is_file():
        return ToolResult(False, f"Not a file: '{relative_path}'")
    suffix = path.suffix.lower()
    allowed = _image_suffixes()
    if suffix not in allowed:
        return ToolResult(
            False,
            f'Unsupported image type {suffix!r}. '
            f'Supported: {", ".join(sorted(allowed))}',
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ToolResult(False, f'Could not read image. {exc}')
    if len(data) > lim.image_max_bytes:
        return ToolResult(
            False,
            f'Image too large ({len(data)} bytes). '
            f'Maximum is {lim.image_max_bytes} bytes.',
        )
    prompt = _vision_prompt(mode)
    if prompt is None:
        valid = ', '.join(sorted(_VISION_MODE_NAMES))
        return ToolResult(False, f"Unknown mode '{mode}'. Valid: {valid}.")
    b64 = base64.b64encode(data).decode('ascii')
    vm = active_vision_model()
    from brain import ui as _ui

    wait = _ui.generation_wait_start(ctx.defaults.ui.cli_vision_wait_message)
    try:
        vr = ctx.ollama_client.chat(
            model=vm,
            messages=[{'role': 'user', 'content': prompt, 'images': [b64]}],
            stream=False,
        )
    except Exception as exc:
        return ToolResult(
            False,
            f'Vision request failed ({vm}). {exc}. '
            'Ensure the model supports images or set --vision-model / '
            'OLLAMA_VISION_MODEL.',
        )
    finally:
        wait.finish()
    msg = msg_to_dict(vr['message'])
    text = (msg.get('content') or '').strip()
    if not text:
        return ToolResult(False, 'Vision model returned empty content.')
    rel = path.relative_to(vault)
    return ToolResult(True, f'[Image: {rel} | model={vm} | mode={mode}]\n\n{text}')


def rename_note(note_title: str, new_title: str) -> ToolResult:
    """
    Rename a note in place (same folder, new filename).

    Automatically updates all [[wikilinks]] pointing to the old title across
    the vault. Use move_note instead if you want to change the folder.
    """
    path = _resolve_note_path(note_title)
    if not path:
        return ToolResult(False, f"Note '{note_title}' not found.")
    vault = _vault()
    dest = path.parent / _ensure_md(new_title)
    try:
        _assert_in_vault(dest)
    except ValueError as exc:
        return ToolResult(False, f'{exc}')
    if dest.exists():
        return ToolResult(False, f"'{new_title}' already exists in the same folder.")
    old_stem = path.stem
    try:
        path.rename(dest)
        _NOTE_INDEX.invalidate()
    except OSError as exc:
        return ToolResult(False, f'Could not rename note. {exc}')
    updated = _update_wikilinks(old_stem, dest.stem)
    suffix = f' Updated [[links]] in {updated} other note(s).' if updated else ''
    return ToolResult(True, f"Renamed '{note_title}' → '{dest.name}'.{suffix}")


def search_by_tag(tag: str) -> ToolResult:
    """
    Find all notes whose YAML frontmatter ``tags`` field contains the given tag.

    Matching is case-insensitive. Tags may be stored as a YAML list
    (``tags: [python, tutorial]``) or a comma-separated string. Pair with
    update_frontmatter to add or change tags.
    """
    if not YAML_AVAILABLE:
        return ToolResult(False, 'PyYAML is not installed. Run: pip install PyYAML')
    vault = _vault()
    lim = get_context().defaults.limits
    tag_lower = tag.strip().lower()
    matches: list[str] = []
    for note_path in _vault_notes():
        try:
            content = _read_vault_text(note_path)
        except OSError:
            continue
        meta, _ = _parse_frontmatter(content)
        tags = meta.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        if not isinstance(tags, list):
            continue
        if any(t.strip().lower() == tag_lower for t in tags if isinstance(t, str)):
            matches.append(f'  📄 {note_path.relative_to(vault)}')
            if len(matches) >= lim.max_search_results:
                break
    if not matches:
        return ToolResult(True, f"No notes found with tag '{tag}'.")
    return ToolResult(
        True,
        f"Notes tagged '{tag}' ({len(matches)}):\n" + '\n'.join(matches),
    )


def _update_wikilinks(old_stem: str, new_stem: str) -> int:
    """Replace [[old_stem]] with [[new_stem]] across all vault notes. Returns files changed."""
    if old_stem == new_stem:
        return 0
    pat = re.compile(
        r'\[\[' + re.escape(old_stem) + r'(\|[^\]]+)?\]\]',
        re.IGNORECASE,
    )
    enc = _encoding()
    changed = 0
    for note_path in _vault_notes():
        try:
            content = _read_vault_text(note_path)
        except OSError:
            continue
        new_content = pat.sub(lambda m: f'[[{new_stem}{m.group(1) or ""}]]', content)
        if new_content != content:
            try:
                note_path.write_text(new_content, encoding=enc)
                changed += 1
            except OSError as exc:
                logger.warning('Could not write wikilink update to %s: %s', note_path, exc)
    return changed


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Tools whose ``content`` argument may be huge — terminal log preview truncates these.
CONTENT_HEAVY_TOOLS: Final[frozenset[str]] = frozenset({'create_note', 'edit_note'})

TOOL_MAP: dict[str, Callable[..., ToolResult]] = {
    'list_directory': list_directory,
    'read_note': read_note,
    'create_note': create_note,
    'edit_note': edit_note,
    'move_note': move_note,
    'rename_note': rename_note,
    'delete_note': delete_note,
    'search_notes': search_notes,
    'search_by_tag': search_by_tag,
    'get_backlinks': get_backlinks,
    'update_frontmatter': update_frontmatter,
    'read_image': read_image,
}

TOOL_LIST: list[Callable[..., ToolResult]] = list(TOOL_MAP.values())
