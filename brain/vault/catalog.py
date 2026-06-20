"""Unified vault catalog: path index + persistent full-text search index."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from brain.core.atomic_io import atomic_write_text
from brain.core.context import get_context
from brain.vault.notes import iter_vault_notes
from brain.vault.paths import assert_in_vault, is_hidden_vault_path, vault_root

logger = logging.getLogger(__name__)

INDEX_FILENAME = '.agent_search_index.json'
INDEX_VERSION = 1

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _index_path() -> Path:
    return vault_root() / INDEX_FILENAME


@dataclass
class NoteRecord:
    """Cached note content keyed by vault-relative path."""

    rel_path: str
    mtime_ns: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    """One full-text search match."""

    rel_path: str
    kind: Literal['filename', 'line']
    line_number: int = 0
    line_text: str = ''


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    if not _YAML_AVAILABLE or not content.startswith('---'):
        return {}, content
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}, content
    try:
        meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, content[match.end():]


class VaultCatalog:
    """
    Per-process vault catalog: case-insensitive filename lookup and
    persistent full-text index (``.agent_search_index.json`` v1).
    """

    def __init__(self) -> None:
        self._notes: dict[str, NoteRecord] = {}
        self._text_loaded: bool = False
        self._path_index: dict[str, list[Path]] = {}
        self._path_index_valid: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path index (lazy rebuild; skipped on content-only edits)
    # ------------------------------------------------------------------

    def _rebuild_path_index_unlocked(self) -> None:
        self._path_index = {}
        for p in iter_vault_notes():
            self._path_index.setdefault(p.name.lower(), []).append(p)
        self._path_index_valid = True

    def _ensure_path_index(self) -> None:
        with self._lock:
            if not self._path_index_valid:
                self._rebuild_path_index_unlocked()

    def find_by_title(self, title: str) -> Path | None:
        """Return the first note matching *title* (case-insensitive filename)."""
        filename = title if title.endswith('.md') else f'{title}.md'
        self._ensure_path_index()
        with self._lock:
            paths = self._path_index.get(filename.lower(), [])
            return paths[0] if paths else None

    def count_notes_under(self, folder: Path) -> int:
        """Count indexed ``.md`` notes under *folder* (non-hidden)."""
        self._ensure_path_index()
        vault = vault_root()
        with self._lock:
            n = 0
            for paths in self._path_index.values():
                for p in paths:
                    try:
                        p.relative_to(folder)
                    except ValueError:
                        continue
                    try:
                        rel = p.relative_to(vault)
                    except ValueError:
                        continue
                    if not is_hidden_vault_path(rel):
                        n += 1
            return n

    def invalidate(self) -> None:
        """Force full reload of path and text indexes on next access."""
        with self._lock:
            self._path_index_valid = False
            self._path_index.clear()
            self._text_loaded = False
            self._notes.clear()

    # ------------------------------------------------------------------
    # Persistent text index
    # ------------------------------------------------------------------

    def _load_disk(self) -> None:
        path = _index_path()
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Search index corrupt or unreadable (%s); rebuilding.', exc)
            return
        if not isinstance(raw, dict) or raw.get('version') != INDEX_VERSION:
            return
        notes = raw.get('notes')
        if not isinstance(notes, dict):
            return
        for rel, entry in notes.items():
            if not isinstance(rel, str) or not isinstance(entry, dict):
                continue
            mtime = entry.get('mtime_ns')
            text = entry.get('text')
            if isinstance(mtime, int) and isinstance(text, str):
                self._notes[rel] = NoteRecord(rel_path=rel, mtime_ns=mtime, text=text)

    def _persist(self) -> None:
        payload = {
            'version': INDEX_VERSION,
            'notes': {
                rel: {'mtime_ns': e.mtime_ns, 'text': e.text}
                for rel, e in sorted(self._notes.items())
            },
        }
        atomic_write_text(
            _index_path(),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _encoding(self) -> str:
        return get_context().user.note_encoding or 'utf-8'

    def _read_note_text(self, abs_path: Path) -> tuple[int, str]:
        text = abs_path.read_text(encoding=self._encoding(), errors='replace')
        stat = abs_path.stat()
        return stat.st_mtime_ns, text

    def _ensure_text_entry(self, abs_path: Path, rel_s: str) -> NoteRecord:
        mtime_ns, text = self._read_note_text(abs_path)
        existing = self._notes.get(rel_s)
        if existing is None or existing.mtime_ns != mtime_ns:
            self._notes[rel_s] = NoteRecord(rel_path=rel_s, mtime_ns=mtime_ns, text=text)
            self._persist()
        return self._notes[rel_s]

    def _scan_vault_text(self) -> None:
        found: dict[str, NoteRecord] = {}
        enc = self._encoding()
        root = vault_root()
        for abs_path in iter_vault_notes():
            rel_s = abs_path.relative_to(root).as_posix()
            mtime_ns = abs_path.stat().st_mtime_ns
            text = abs_path.read_text(encoding=enc, errors='replace')
            found[rel_s] = NoteRecord(rel_path=rel_s, mtime_ns=mtime_ns, text=text)
        self._notes = found
        self._persist()

    def _ensure_text_loaded(self) -> None:
        with self._lock:
            if self._text_loaded:
                return
            self._load_disk()
            root = vault_root()
            dirty = False
            for abs_path in iter_vault_notes():
                rel_s = abs_path.relative_to(root).as_posix()
                mtime_ns = abs_path.stat().st_mtime_ns
                entry = self._notes.get(rel_s)
                if entry is None or entry.mtime_ns != mtime_ns:
                    try:
                        text = abs_path.read_text(
                            encoding=self._encoding(),
                            errors='replace',
                        )
                    except OSError:
                        continue
                    self._notes[rel_s] = NoteRecord(
                        rel_path=rel_s,
                        mtime_ns=mtime_ns,
                        text=text,
                    )
                    dirty = True
            stale = [rel for rel in self._notes if not (root / rel).is_file()]
            for rel in stale:
                del self._notes[rel]
                dirty = True
            if dirty or not self._notes:
                if not self._notes and not _index_path().is_file():
                    self._scan_vault_text()
                elif dirty:
                    self._persist()
            self._text_loaded = True

    def _rel_key(self, abs_path: Path) -> str | None:
        root = vault_root()
        try:
            rel = abs_path.relative_to(root)
        except ValueError:
            return None
        if is_hidden_vault_path(rel):
            return None
        return rel.as_posix()

    def _remove_text_entry(self, abs_path: Path) -> None:
        rel_s = self._rel_key(abs_path)
        if rel_s is None:
            return
        with self._lock:
            self._text_loaded = True
            if rel_s in self._notes:
                del self._notes[rel_s]
                self._persist()

    def _update_text_entry(self, abs_path: Path) -> None:
        rel_s = self._rel_key(abs_path)
        if rel_s is None or not abs_path.is_file():
            return
        with self._lock:
            self._text_loaded = True
            try:
                real = assert_in_vault(abs_path)
                self._ensure_text_entry(real, rel_s)
            except (ValueError, OSError):
                pass

    def on_note_changed(
        self,
        path: Path,
        *,
        path_changed: bool = False,
        deleted: bool = False,
        former_path: Path | None = None,
    ) -> None:
        """
        Refresh indexes after a vault mutation.

        Content-only edits skip path-index rebuild; creates, moves, renames,
        and deletes set ``path_changed`` / ``deleted`` / ``former_path``.
        """
        if former_path is not None:
            self._remove_text_entry(former_path)
        if deleted:
            self._remove_text_entry(path)
            with self._lock:
                self._path_index_valid = False
            return
        if path_changed:
            with self._lock:
                self._path_index_valid = False
        if path.is_file():
            self._update_text_entry(path)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        *,
        case_sensitive: bool = False,
    ) -> list[SearchHit]:
        """Full-text search across note bodies and filename stems."""
        self._ensure_text_loaded()
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)
        hits: list[SearchHit] = []
        with self._lock:
            records = sorted(self._notes.values(), key=lambda r: r.rel_path)
        for record in records:
            rel_path = Path(record.rel_path)
            if pattern.search(rel_path.stem):
                hits.append(SearchHit(rel_path=record.rel_path, kind='filename'))
                continue
            for i, line in enumerate(record.text.splitlines(), 1):
                if pattern.search(line):
                    hits.append(
                        SearchHit(
                            rel_path=record.rel_path,
                            kind='line',
                            line_number=i,
                            line_text=line.strip(),
                        ),
                    )
        return hits

    def backlinks(self, note_title: str) -> list[str]:
        """Return vault-relative paths of notes that wikilink to *note_title*."""
        target = note_title.removesuffix('.md')
        pat = re.compile(
            r'\[\[' + re.escape(target) + r'(?:\|[^\]]+)?\]\]',
            re.IGNORECASE,
        )
        self._ensure_text_loaded()
        matches: list[str] = []
        with self._lock:
            records = sorted(self._notes.values(), key=lambda r: r.rel_path)
        for record in records:
            if pat.search(record.text):
                matches.append(record.rel_path)
        return matches

    def search_tag(self, tag: str) -> list[str]:
        """Return vault-relative paths whose frontmatter ``tags`` contain *tag*."""
        if not _YAML_AVAILABLE:
            return []
        tag_lower = tag.strip().lower()
        self._ensure_text_loaded()
        matches: list[str] = []
        with self._lock:
            records = sorted(self._notes.values(), key=lambda r: r.rel_path)
        for record in records:
            meta, _ = _parse_frontmatter(record.text)
            tags = meta.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            if not isinstance(tags, list):
                continue
            if any(
                t.strip().lower() == tag_lower
                for t in tags
                if isinstance(t, str)
            ):
                matches.append(record.rel_path)
        return matches


_CATALOG = VaultCatalog()


def get_vault_catalog() -> VaultCatalog:
    """Return the process-wide vault catalog singleton."""
    return _CATALOG
