"""Second Brain agent — modular vault + Ollama assistant."""

from brain.agent_loop import run_agent
from brain.bootstrap import bootstrap

__all__ = ['bootstrap', 'run_agent']
