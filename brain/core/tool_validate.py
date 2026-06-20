"""Pre-flight validation for Ollama tool calls."""

from __future__ import annotations

from typing import Any

from brain.core.context import get_context

_VAULT_RELATIVE_TOOLS: frozenset[str] = frozenset({
    'list_directory',
    'create_folder',
    'delete_folder',
    'read_image',
})

_NOTE_TITLE_TOOLS: frozenset[str] = frozenset({
    'read_note',
    'create_note',
    'edit_note',
    'move_note',
    'rename_note',
    'delete_note',
    'get_backlinks',
    'update_frontmatter',
})

_EDIT_MODES: frozenset[str] = frozenset({'append', 'prepend', 'overwrite'})
_VISION_MODES: frozenset[str] = frozenset({'ocr', 'describe', 'full'})


def _max_arg_chars() -> int:
    return get_context().defaults.limits.max_tool_arg_chars


def _reject_vault_relative(value: str, field: str) -> str | None:
    if not isinstance(value, str):
        return f'{field} must be a string.'
    if len(value) > _max_arg_chars():
        return f'{field} exceeds maximum length ({_max_arg_chars()} characters).'
    stripped = value.strip()
    if value != stripped and stripped == '':
        return f'{field} must not be whitespace only.'
    normalized = value.strip().replace('\\', '/')
    if normalized.startswith('/'):
        return f'{field} must be vault-relative, not an absolute path.'
    parts = [p for p in normalized.split('/') if p]
    if '..' in parts:
        return f'{field} must not contain parent-directory segments (..).'
    return None


def _require_non_empty_str(args: dict[str, Any], field: str) -> str | None:
    val = args.get(field)
    if not isinstance(val, str):
        return f'{field} must be a non-empty string.'
    if not val.strip():
        return f'{field} must be a non-empty string.'
    if len(val) > _max_arg_chars():
        return f'{field} exceeds maximum length ({_max_arg_chars()} characters).'
    return None


def validate_tool_call(name: str, args: dict[str, Any]) -> str | None:
    """
    Validate tool arguments before dispatch.

    Returns an error string for the model, or ``None`` if OK.
    """
    if not isinstance(args, dict):
        return 'Tool arguments must be a JSON object.'

    if name in _NOTE_TITLE_TOOLS:
        err = _require_non_empty_str(args, 'note_title')
        if err:
            return err

    if name == 'read_note':
        start = args.get('start_line', 1)
        max_lines = args.get('max_lines', 0)
        if not isinstance(start, int) or isinstance(start, bool):
            return 'start_line must be an integer.'
        if not isinstance(max_lines, int) or isinstance(max_lines, bool):
            return 'max_lines must be an integer.'
        if start < 1:
            return 'start_line must be >= 1.'
        if max_lines < 0:
            return 'max_lines must be >= 0.'

    if name == 'create_folder':
        err = _require_non_empty_str(args, 'relative_path')
        if err:
            return err
        rel_err = _reject_vault_relative(str(args.get('relative_path', '')), 'relative_path')
        if rel_err:
            return rel_err

    if name in {'delete_folder', 'list_directory', 'read_image'}:
        field = 'relative_path'
        val = args.get(field, '')
        if name != 'list_directory' or (isinstance(val, str) and val.strip()):
            if name in {'delete_folder', 'read_image'}:
                err = _require_non_empty_str(args, field)
                if err:
                    return err
            if isinstance(val, str) and val.strip():
                rel_err = _reject_vault_relative(val, field)
                if rel_err:
                    return rel_err

    if name in _VAULT_RELATIVE_TOOLS and name not in {'create_folder', 'delete_folder', 'read_image'}:
        val = args.get('relative_path', '')
        if isinstance(val, str) and val.strip():
            rel_err = _reject_vault_relative(val, 'relative_path')
            if rel_err:
                return rel_err

    if name == 'search_notes':
        err = _require_non_empty_str(args, 'query')
        if err:
            return err

    if name == 'search_by_tag':
        err = _require_non_empty_str(args, 'tag')
        if err:
            return err

    if name == 'edit_note':
        mode = args.get('mode', 'append')
        if not isinstance(mode, str) or mode not in _EDIT_MODES:
            valid = ', '.join(sorted(_EDIT_MODES))
            return f"mode must be one of: {valid}."

    if name == 'read_image':
        mode = args.get('mode', 'full')
        if not isinstance(mode, str) or mode not in _VISION_MODES:
            valid = ', '.join(sorted(_VISION_MODES))
            return f"mode must be one of: {valid}."

    if name in {'delete_note', 'delete_folder'}:
        confirm = args.get('confirm', False)
        if confirm is not True:
            return (
                f'{name} requires confirm=true (boolean true) after explicit user '
                'confirmation.'
            )

    if name == 'update_frontmatter':
        meta = args.get('metadata')
        if not isinstance(meta, dict):
            return 'metadata must be a JSON object (dict).'

    if name == 'rename_note':
        err = _require_non_empty_str(args, 'new_title')
        if err:
            return err

    if name == 'create_note':
        content = args.get('content')
        if not isinstance(content, str):
            return 'content must be a string.'
        folder = args.get('folder', '')
        if folder is not None and not isinstance(folder, str):
            return 'folder must be a string.'
        if isinstance(folder, str) and folder.strip():
            rel_err = _reject_vault_relative(folder, 'folder')
            if rel_err:
                return rel_err

    if name == 'move_note':
        dest = args.get('destination_folder', '')
        if not isinstance(dest, str):
            return 'destination_folder must be a string.'
        if dest.strip():
            rel_err = _reject_vault_relative(dest, 'destination_folder')
            if rel_err:
                return rel_err

    return None
