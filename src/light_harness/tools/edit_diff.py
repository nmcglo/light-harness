"""Text-matching and diff helpers for :class:`~light_harness.tools.builtin.EditFileTool`.

This mirrors the semantics of Pi's ``edit`` tool: multiple disjoint exact-text
replacements per call, all matched against the *original* file (not
incrementally), with an exact-then-fuzzy fallback and preservation of the BOM,
the original line endings, and the bytes of unchanged lines.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# A located edit: (index, match_length, new_text, edit_index).
MatchedEdit = tuple[int, int, str, int]


class EditError(Exception):
    """Raised when an edit cannot be applied; the message is shown to the model."""


def split_bom(raw: str) -> tuple[str, str]:
    """Split a leading UTF-8 BOM from ``raw``, returning ``(bom, text)``."""
    if raw.startswith("\ufeff"):
        return "\ufeff", raw[1:]
    return "", raw


def detect_line_ending(content: str) -> str:
    """Return ``"\\r\\n"`` when CRLF occurs first, otherwise ``"\\n"``."""
    crlf = content.find("\r\n")
    lf = content.find("\n")
    if lf == -1:
        return "\n"
    if crlf == -1:
        return "\n"
    return "\r\n" if crlf < lf else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize every line ending to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Convert ``\\n`` back to ``ending`` (a no-op when ``ending`` is ``"\\n"``)."""
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching: NFKC, per-line trailing whitespace, and
    smart quotes / dashes / special spaces mapped to their ASCII equivalents."""
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    text = re.sub(r"[\u00a0\u2002-\u200a\u202f\u205f\u3000]", " ", text)
    return text


def _fuzzy_find(content: str, old: str) -> tuple[bool, int, int, bool, str]:
    """Locate ``old`` in ``content``: exact match first, then fuzzy.

    Returns ``(found, index, match_length, used_fuzzy, content_for_replacement)``.
    When fuzzy is used, ``index``/``match_length`` refer to the fuzzy-normalized
    content returned as ``content_for_replacement``.

    A pattern whose fuzzy form is empty never matches: ``find("")`` returns 0
    rather than -1, which would read as a zero-length match at byte 0 and let a
    replacement be prepended to the file.
    """
    fuzzy_old = normalize_for_fuzzy_match(old)
    if not fuzzy_old:
        return False, -1, 0, False, content
    exact = content.find(old)
    if exact != -1:
        return True, exact, len(old), False, content
    fuzzy_content = normalize_for_fuzzy_match(content)
    index = fuzzy_content.find(fuzzy_old)
    if index == -1:
        return False, -1, 0, False, content
    return True, index, len(fuzzy_old), True, fuzzy_content


def _count_occurrences(content: str, old: str) -> int:
    """Count (fuzzy) occurrences of ``old`` in ``content``.

    ``old`` must have a non-empty fuzzy form; an empty pattern splits at every
    position, so its occurrence count is meaningless.
    """
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old)
    return len(fuzzy_content.split(fuzzy_old)) - 1


def _empty(path: str, index: int, total: int) -> str:
    if total == 1:
        return f"old_text must not be empty or whitespace-only in {path}."
    return f"edits[{index}].old_text must not be empty or whitespace-only in {path}."


def _not_found(path: str, index: int, total: int) -> str:
    if total == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{index}] in {path}. The old_text must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate(path: str, index: int, total: int, occurrences: int) -> str:
    if total == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must be "
            "unique. Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{index}] in {path}. Each old_text "
        "must be unique. Please provide more context to make it unique."
    )


def _no_change(path: str, total: int) -> str:
    if total == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate special characters or the text not existing as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def _apply_replacements(
    content: str, replacements: list[MatchedEdit], offset: int = 0
) -> str:
    """Apply ``replacements`` in reverse order so earlier offsets stay stable."""
    result = content
    for index, length, new_text, _edit_index in reversed(replacements):
        at = index - offset
        result = result[:at] + new_text + result[at + length:]
    return result


def _split_lines_with_endings(content: str) -> list[str]:
    """Split keeping each line's trailing newline (final line may lack one)."""
    lines: list[str] = []
    i = 0
    n = len(content)
    while i < n:
        j = content.find("\n", i)
        if j == -1:
            lines.append(content[i:])
            break
        lines.append(content[i : j + 1])
        i = j + 1
    return lines


def _line_spans(lines: list[str]) -> list[tuple[int, int]]:
    """``(start, end)`` byte offsets of each line within the joined content."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in lines:
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _replacement_line_range(
    spans: list[tuple[int, int]], index: int, length: int
) -> tuple[int, int]:
    """Return ``(start_line, end_line_exclusive)`` of the lines a replacement touches."""
    start = index
    end = index + length
    start_line = -1
    for i, (line_start, line_end) in enumerate(spans):
        if line_start <= start < line_end:
            start_line = i
            break
    if start_line == -1:
        raise EditError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(spans) and spans[end_line][1] < end:
        end_line += 1
    if end_line >= len(spans):
        raise EditError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_preserving_lines(
    original: str, base: str, replacements: list[MatchedEdit]
) -> str:
    """Apply replacements made in ``base`` (a normalized view) to ``original``.

    Each replacement is widened to the lines it touches; those lines are rewritten
    from ``base`` (with the replacements applied) and all other lines are copied
    verbatim from ``original``. Only lines outside every touched group keep their
    exact bytes: a touched line also takes on the fuzzy normalization of ``base``
    (smart quotes to ASCII, trailing whitespace stripped).
    """
    original_lines = _split_lines_with_endings(original)
    base_spans = _line_spans(_split_lines_with_endings(base))
    if len(original_lines) != len(base_spans):
        raise EditError("Cannot preserve unchanged lines: line counts differ.")

    sorted_reps = sorted(replacements, key=lambda r: r[0])
    groups: list[tuple[int, int, list[MatchedEdit]]] = []
    for rep in sorted_reps:
        start_line, end_line = _replacement_line_range(base_spans, rep[0], rep[1])
        if groups and start_line < groups[-1][1]:
            groups[-1] = (groups[-1][0], max(groups[-1][1], end_line), groups[-1][2] + [rep])
        else:
            groups.append((start_line, end_line, [rep]))

    result: list[str] = []
    original_index = 0
    for start_line, end_line, reps in groups:
        result.append("".join(original_lines[original_index:start_line]))
        group_start = base_spans[start_line][0]
        group_end = base_spans[end_line - 1][1]
        result.append(_apply_replacements(base[group_start:group_end], reps, group_start))
        original_index = end_line
    result.append("".join(original_lines[original_index:]))
    return "".join(result)


def apply_edits(
    normalized_content: str, edits: list[tuple[str, str]], path: str
) -> tuple[str, str]:
    """Apply ``edits`` (pairs of ``(old_text, new_text)``) to LF-normalized content.

    ``normalized_content`` must already be LF-normalized with the BOM stripped.
    Returns ``(base_content, new_content)``. Raises :class:`EditError` on any
    validation problem (empty or whitespace-only, not-found, non-unique,
    overlapping, no-op).
    """
    total = len(edits)
    normalized = [(normalize_to_lf(o), normalize_to_lf(n)) for o, n in edits]
    for i, (o, _n) in enumerate(normalized):
        # A pattern made only of spaces, tabs or special spaces has an empty
        # fuzzy form, so it has no region to replace; accepting it would match
        # at byte 0 with zero length and prepend new_text to the file.
        if not o or not normalize_for_fuzzy_match(o):
            raise EditError(_empty(path, i, total))

    # First pass decides whether any edit needs fuzzy matching; if so, all
    # matching/replacement happens in fuzzy-normalized space.
    first_pass = [_fuzzy_find(normalized_content, o) for o, _n in normalized]
    used_fuzzy = any(fp[3] for fp in first_pass)
    replacement_base = (
        normalize_for_fuzzy_match(normalized_content) if used_fuzzy else normalized_content
    )

    matched: list[MatchedEdit] = []
    for i, (o, n) in enumerate(normalized):
        found, index, match_len, matched_fuzzy, _cfr = _fuzzy_find(replacement_base, o)
        if not found:
            raise EditError(_not_found(path, i, total))
        # An exact hit is counted exactly: counting in fuzzy space would call
        # "hello " ambiguous in "hello \nhello\n" although it occurs once.
        occurrences = (
            _count_occurrences(replacement_base, o)
            if matched_fuzzy
            else replacement_base.count(o)
        )
        if occurrences > 1:
            raise EditError(_duplicate(path, i, total, occurrences))
        matched.append((index, match_len, n, i))

    # Decided here, on the matched spans, because once fuzzy matching is in play
    # the rewritten lines are never byte-equal to the original (smart quotes,
    # trailing whitespace), so comparing the final text could not detect a no-op.
    if all(replacement_base[index : index + length] == new for index, length, new, _ in matched):
        raise EditError(_no_change(path, total))

    matched.sort(key=lambda m: m[0])
    for i in range(1, len(matched)):
        prev = matched[i - 1]
        cur = matched[i]
        if prev[0] + prev[1] > cur[0]:
            raise EditError(
                f"edits[{prev[3]}] and edits[{cur[3]}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    if used_fuzzy:
        new_content = _apply_preserving_lines(normalized_content, replacement_base, matched)
    else:
        new_content = _apply_replacements(replacement_base, matched)

    return base_content, new_content


def generate_diff(old: str, new: str, path: str) -> str:
    """A unified diff between ``old`` and ``new`` (empty when identical)."""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)
