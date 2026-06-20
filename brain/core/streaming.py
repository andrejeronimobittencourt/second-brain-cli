"""Helpers for merging streamed LLM content chunks."""


def merge_stream_content(previous: str, piece: str) -> str:
    """
    Merge one streamed ``content`` field into running assistant text.

    Ollama and other providers may emit either token deltas or the full text
    accumulated so far; this helper accepts both without double-counting.
    """
    if not piece:
        return previous
    if not previous:
        return piece
    if piece == previous:
        return previous
    if piece.startswith(previous):
        return piece
    if previous.startswith(piece):
        return previous
    return previous + piece
