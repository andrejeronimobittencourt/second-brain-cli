"""
Application defaults.

All tuning, copy, and behaviour live here as immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


def _latex_pairs_from_map(m: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(m.items(), key=lambda kv: len(kv[0]), reverse=True))


_BASE_LATEX: Final[dict[str, str]] = {
    r'\Leftrightarrow': '⇔',
    r'\leftrightarrow': '↔',
    r'\Rightarrow': '⇒',
    r'\rightarrow': '→',
    r'\Leftarrow': '⇐',
    r'\leftarrow': '←',
    r'\times': '×',
    r'\cdot': '·',
    r'\pm': '±',
    r'\mp': '∓',
    r'\leq': '≤',
    r'\geq': '≥',
    r'\neq': '≠',
    r'\approx': '≈',
    r'\infty': '∞',
    r'\sum': 'Σ',
    r'\prod': 'Π',
    r'\sqrt': '√',
    r'\alpha': 'α',
    r'\beta': 'β',
    r'\gamma': 'γ',
    r'\delta': 'δ',
    r'\pi': 'π',
    r'\sigma': 'σ',
    r'\theta': 'θ',
    r'\lambda': 'λ',
    r'\mu': 'μ',
    r'\omega': 'ω',
    r'\Delta': 'Δ',
    r'\Omega': 'Ω',
    r'\ldots': '…',
    r'\cdots': '⋯',
}


@dataclass(frozen=True)
class Limits:
    max_context_messages: int = 40
    max_tool_rounds: int = 15
    max_search_results: int = 20
    max_matches_per_file: int = 3
    max_list_other_files: int = 40
    max_history_messages: int = 200
    image_max_bytes: int = 25 * 1024 * 1024
    search_snippet_chars: int = 120
    # ``create_note`` / ``edit_note`` log only this many characters of the first line.
    tool_call_content_preview_chars: int = 80


@dataclass(frozen=True)
class MarkdownStyle:
    ruler_min_repeated_chars: int = 24
    ruler_charset_chars: str = '-_*'


@dataclass(frozen=True)
class TerminalStyle:
    panel_min_outer_width: int = 40
    panel_fallback_outer_width: int = 88
    panel_inner_margin_columns: int = 4
    # Extra columns subtracted from the Constrain width inside panels (headroom for
    # list gutters, bold markers, and other Markdown chrome that adds visual width).
    panel_prewrap_shrink_columns: int = 4
    # Subtract from measured columns (IDE / scrollbars often report width +1).
    panel_width_safety_columns: int = 1
    prewrap_min_inner_width: int = 20
    hard_wrap_min_line_length: int = 16
    hard_wrap_space_break_min_fraction: float = 0.66
    rich_soft_wrap: bool = True
    print_crop: bool = False
    print_overflow: str = 'fold'
    # Rich ``Status`` spinner name while waiting for the first streamed chunk.
    cli_generation_spinner: str = 'dots'


@dataclass(frozen=True)
class VisionPrompts:
    ocr: str = (
        'Transcribe all visible text in this image accurately (OCR). '
        'Preserve line breaks where meaningful. If there is no text, say so briefly.'
    )
    describe: str = (
        'Describe this image for note-taking: main subjects, layout, colours, '
        'and any readable text worth mentioning.'
    )
    full: str = (
        'For a vault note: (1) Transcribe all visible text (OCR). '
        '(2) Briefly describe non-text content if it matters for context. '
        'Use the same language as the text in the image when possible.'
    )


@dataclass(frozen=True)
class ModelStyle:
    channel_leak_regex: str = r'<channel[^>]*>'
    empty_answer_markdown: str = (
        '_(The model returned **empty** answer text—common right after a tool step '
        'if it stops early. Ask again, or use `/clear`.)_\n\n'
        '**Note:** `read_note` accepts vault-relative paths (e.g. '
        '`Subject/Topic/NoteName`), not only bare titles.'
    )


@dataclass(frozen=True)
class UIStyle:
    panel_answer_title: str = 'Second Brain'
    panel_reasoning_title: str = 'Reasoning'
    cli_generation_wait_message: str = 'Working…'
    cli_generation_wait_message_think: str = 'Thinking…'
    cli_vision_wait_message: str = 'Vision model…'
    cli_compression_wait_message: str = 'Compressing prior context…'
    rich_theme: tuple[tuple[str, str], ...] = (
        ('agent.name', 'bold cyan'),
        ('agent.tool', 'bold yellow'),
        ('agent.error', 'bold red'),
        ('agent.info', 'dim white'),
    )


@dataclass(frozen=True)
class FilePolicy:
    image_extensions: tuple[str, ...] = (
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif',
    )


@dataclass(frozen=True)
class ContextCompression:
    """
    When non-system context exceeds ``Limits.max_context_messages``, older
    turns are summarized into one compact assistant message instead of being
    discarded outright (lazy / rolling summary).
    """

    enabled: bool = True
    max_summary_lines: int = 8
    max_transcript_chars_per_message: int = 6000
    summary_message_label: str = (
        '[Prior context — compressed summary; details may be incomplete]'
    )
    summarizer_system_prompt: str = (
        'You compress earlier chat turns for a Markdown vault assistant. '
        'Write up to the requested maximum number of lines as concise plain text '
        '(no markdown code fences). '
        'Put the most recent user intent on the first line. '
        'Then include: note paths or titles read, created, edited, moved, or searched; '
        'important tool outcomes; decisions or constraints the user stated. '
        'Copy note paths and titles exactly as they appear — do not paraphrase or '
        'abbreviate them, as the assistant will use them for future tool calls. '
        'Do not invent paths, titles, or quotes. If something is uncertain, omit it.'
    )


@dataclass(frozen=True)
class AppDefaults:
    """Single aggregate for all built-in behaviour."""

    limits: Limits = field(default_factory=Limits)
    markdown: MarkdownStyle = field(default_factory=MarkdownStyle)
    terminal: TerminalStyle = field(default_factory=TerminalStyle)
    vision: VisionPrompts = field(default_factory=VisionPrompts)
    model: ModelStyle = field(default_factory=ModelStyle)
    ui: UIStyle = field(default_factory=UIStyle)
    files: FilePolicy = field(default_factory=FilePolicy)
    context_compression: ContextCompression = field(default_factory=ContextCompression)
    latex_symbol_pairs: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: _latex_pairs_from_map(dict(_BASE_LATEX)),
    )


APP_DEFAULTS: Final[AppDefaults] = AppDefaults()


def default_system_prompt() -> str:
    """Built-in system prompt when the user does not override ``system_prompt`` in JSON."""
    return (
        'You are an autonomous assistant for a local Markdown vault (a folder of `.md` '
        'notes and related files) — a local knowledge base — backed by the user\'s '
        'local Ollama model.\n\n'
        '## Vault Structure\n'
        'The vault contains subject folders. Notes use Markdown with optional YAML '
        'frontmatter (tags, dates, aliases) and [[wikilinks]] to interconnect ideas.\n\n'
        '## Markdown links the user can click\n'
        '- Inline links use `[label](destination)`. Many parsers treat an unescaped '
        'space in `destination` as the end of the URL, so the link breaks even though '
        'the file exists. When the destination is a vault path that contains spaces '
        'or other reserved characters, percent-encode them in the link target only; '
        'on-disk names stay as the filesystem has them.\n'
        '- When the user\'s app resolves notes by `[[wikilinks]]` title, that form '
        'often survives folder and spacing issues better than pasting a long relative '
        'path inside parentheses—use whichever convention matches how their vault '
        'already links notes.\n\n'
        '## Images\n'
        '- list_directory lists image filenames in that folder as a dedicated section; '
        'combine with the folder path for read_image (e.g. attachments/screenshot.png).\n'
        '- Use read_image on vault-relative paths to OCR or describe images with the '
        'vision model. Choose mode=ocr for text-only, describe for a scene summary, '
        'or full for OCR plus brief context.\n\n'
        '## Tool Selection\n'
        'Choose the right tool for each situation — do not default to list_directory '
        'when a faster tool will do:\n'
        '- **Known title or path** → read_note directly.\n'
        '- **Keyword or topic search** → search_notes (also matches filenames).\n'
        '- **Tag-based lookup** → search_by_tag (YAML frontmatter tags).\n'
        '- **Find what links to a note** → get_backlinks (reads every file — use sparingly).\n'
        '- **Unknown vault layout** → list_directory to orient, then read_note or search_notes.\n'
        '- **Rename a note in place** → rename_note (updates all [[wikilinks]] automatically).\n'
        '- **Move to another folder** → move_note.\n\n'
        '## Behaviour Rules\n'
        '- Do not re-list a directory or re-read a note whose content is already visible '
        'in this conversation — use what you already know.\n'
        '- read_note accepts a vault-relative path (e.g. Subject/Topic/NoteName) '
        'or a bare note title; paths must match list_directory output.\n'
        '- Follow [[wikilinks]] with read_note only when the linked note is directly '
        'relevant to answering; do not follow every link unconditionally.\n'
        '- When ending a turn after tools, always write a non-empty answer for the user '
        '(summarise tool results); never stop with only an empty message.\n'
        '- When creating a note, always place it in the most appropriate existing folder; '
        'use list_directory first if you do not yet know the folder structure.\n'
        '- When editing, prefer append mode; only overwrite when a full rewrite is justified.\n'
        '- Never delete a note unless the user has explicitly confirmed the action.\n'
        '- After any write operation, briefly summarise what you did and why.\n'
        '- If a tool call fails, diagnose the cause (wrong path, wrong arguments) and '
        'correct it before retrying — do not repeat the identical failing call.\n'
        '- Answer in the same language the user writes in.\n'
        '- Only ask a clarifying question when ambiguity could cause a destructive or '
        'hard-to-reverse action (e.g. deleting or overwriting the wrong note). '
        'For read-only or easily undoable requests, make a reasonable assumption and proceed.\n'
    )
