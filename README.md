<h1 align="center">🧠 Second Brain CLI</h1>

<p align="center">
  <img src="assets/logo.png" alt="Second Brain CLI" width="280">
</p>

<p align="center">
  <b>Chat with your Markdown vault using a local AI.</b><br>
  <i>No cloud. No API keys. Your notes stay on your machine.</i>
</p>

<p align="center">
  <img src="assets/demo.gif" alt="Second Brain CLI demo — chat with a local vault via Ollama" width="720">
</p>

<p align="center">
  <a href="#requirements">Requirements</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#configuration-second_brain_userjson">Configuration</a> •
  <a href="#slash-commands-repl">Commands</a>
</p>

---

## Why this exists

Most AI note tools want your data in their cloud. **Second Brain CLI** brings the AI to your local files instead — no subscriptions, no data leaving your machine.

Point it at any directory of `.md` files — **Obsidian**, Logseq, Zettlr, or a plain `git` repo — and start a conversation. The model can navigate your folder structure, read and edit notes, search by text or tag, follow wikilinks, and describe images, using whatever [Ollama](https://ollama.com/) model you already have pulled.

## Features

- **🔒 Fully local** — powered by Ollama; no API keys, no cloud sync.
- **🛠️ Vault tools** — list folders, read/create/edit/move/rename/delete notes, search, tags, backlinks, frontmatter, and image OCR/describe.
- **💾 Session persistence** — `--resume` continues where you left off; history is stored in your vault.
- **🧠 Long sessions** — older chat turns are summarised when context fills up.
- **💻 Terminal UI** — scrollable transcript, streaming answers, multiline input, and a status bar.

## Requirements

- **Python** 3.10 or newer (3.12+ recommended).
- **[Ollama](https://ollama.com/)** installed and running, with at least one **chat** model pulled (e.g. `ollama pull gemma4`).
- Python packages listed in **`requirements.txt`**: `ollama` (required), `rich` and `prompt_toolkit` (recommended for the interactive REPL — line editing, history, streaming answers), `PyYAML` (optional — only needed for `update_frontmatter` and `search_by_tag`).

## Quick start

1. **Clone** this repository and enter the project directory.

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Copy** `second_brain_user.example.json` to **`second_brain_user.json`** next to `agent.py`.

4. **Edit** `second_brain_user.json`: set **`vault_path`** to the root folder of your notes (the directory that contains your `.md` files and any subfolders you use). Set **`ollama_model`** to a model you have pulled.

5. **Start Ollama**, then run:

   ```bash
   python agent.py
   ```

If `second_brain_user.json` is missing, the app still starts but uses factory defaults and may treat the script directory as the vault until you configure `vault_path`.

## Configuration (`second_brain_user.json`)

| Key | Description |
| --- | --- |
| `vault_path` | Absolute or user-relative path to your Markdown vault root **(required for real use)**. |
| `ollama_host` | Ollama API base URL (default `http://127.0.0.1:11434`). |
| `ollama_model` | Default chat model tag (e.g. `gemma4:latest`; use whatever you have pulled). |
| `ollama_vision_model` | Optional; if empty, `read_image` falls back to `ollama_model` (see below). |
| `history_filename` | Session file name (default `.agent_history.json`, resolved under the vault path). |
| `note_encoding` | Text encoding for note I/O (default `utf-8`). |
| `system_prompt` | Optional. Replaces the built-in system prompt when set. |
| `vault_instructions` | Optional. Extra rules appended each session (vault layout, link style, etc.). |
| `log_level` | e.g. `DEBUG`, `INFO`, `WARNING` (default effective level is `WARNING`). |
| `log_file` | Optional path for `brain` logger file output. |

Keys starting with **`_`** are ignored by the loader (handy for comments in JSON).

### Customising the model's behaviour with `vault_instructions`

`vault_instructions` is appended to the system prompt every session. Use it to tell the model how *your* vault is organised so it doesn't have to guess:

```json
"vault_instructions": "My vault uses Zettelkasten. All permanent notes live in Zettelkasten/ with date-prefixed filenames (e.g. 2024-03-15 Concept Name). Fleeting notes go in Inbox/. When creating a note, always ask which folder it belongs in if it isn't obvious. Tags follow the format #topic/subtopic."
```

You can also use it to name conventions, set the preferred link style (`[[wikilinks]]` vs `[label](path)`), or restrict what the model may and may not do in your vault.

**Config file location** (first match wins if you pass `--config`):

1. Path given with **`--config`**
2. Environment variable **`SECOND_BRAIN_USER_CONFIG`**
3. **`./second_brain_user.json`** next to `agent.py`

## Command-line interface

```text
python agent.py [--config FILE] [--resume] [--think] [--vision-model MODEL]
                  [--host URL] [--model NAME] [--session NAME] [--print "prompt"]
```

| Flag | Purpose |
| --- | --- |
| `--config` | Path to your user JSON. |
| `--resume` | Load prior non-system messages from the session file. |
| `--think` | Show model reasoning before each answer (requires a thinking-capable Ollama model). |
| `--vision-model` | Override the model used for `read_image` for this run only. |
| `--host` | Override `ollama_host` for this run. |
| `--model` | Override `ollama_model` for this run. |
| `--session` | Use a named session under `{vault}/.agent_sessions/` instead of the default history file. |
| `--print` | Run one prompt and print the answer to stdout (no interactive REPL). |

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SECOND_BRAIN_USER_CONFIG` | Path to `second_brain_user.json` when not using `--config`. |
| `OLLAMA_VISION_MODEL` | Highest-priority override for the vision model used by `read_image`. |

Precedence for the vision model is: **`--vision-model`** / **`/vision-model`** → **`OLLAMA_VISION_MODEL`** → **`ollama_vision_model`** in JSON → **`ollama_model`**. Leaving `ollama_vision_model` empty is fine if your **main** model supports images.

## Slash commands (REPL)

**Input:** Enter to send; Esc Enter or Ctrl+J for a new line; Ctrl+C or Ctrl+D to exit. Scroll the transcript with Page Up/Down, Shift+Up/Down, Ctrl+Up/Down, Home, or End.

**Panels:** answers stream in the **Second Brain** panel; with `--think`, **Reasoning** streams first. `/stats`, `/help`, and other slash commands use dim panels.

| Command | Action |
| --- | --- |
| `/help` | Commands and key bindings |
| `/clear` | Clear history and delete the session file |
| `/history` | Message count in rolling context |
| `/stats` | Context fill, model, session path |
| `/compact` | Compress older turns now |
| `/search <term>` | Vault text search (no LLM) |
| `/model <name>` | Switch chat model |
| `/vision-model <name>` | Switch vision model |
| `/exit` | Save and exit (`/quit`, `/bye`) |

### Scripting

```bash
python agent.py --print "Summarize all notes in Inbox/"
```

Prints the final answer to stdout and exits — useful for shell scripts or Obsidian automation.

## Good to know

- The assistant only reads and writes files **inside** your configured vault.