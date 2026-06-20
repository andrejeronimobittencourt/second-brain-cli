"""Second Brain agent — modular vault + Ollama assistant."""

import brain.ui as ui
from brain.agent.loop import run_agent
from brain.core.bootstrap import bootstrap

__all__ = ['bootstrap', 'run_agent', 'ui']
