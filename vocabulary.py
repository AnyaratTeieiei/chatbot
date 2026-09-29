"""Curated Thai/English aliases; longest phrase wins without recursive expansion."""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

SYNONYM_GROUPS = json.loads(Path(__file__).with_name('synonyms.json').read_text(encoding='utf-8'))


def clean_question(text):
    text = unicodedata.normalize('NFC', text).replace('\u0e4d\u0e32', '\u0e33').translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789'))
    return ' '.join(re.sub('[\u200b\ufeff]', '', text).split())


_aliases = {}
for canonical, aliases in SYNONYM_GROUPS.items():
    for alias in [canonical, *aliases]:
        key = clean_question(alias).casefold()
        if key in _aliases and _aliases[key] != canonical:
            raise ValueError(f'Conflicting synonym: {alias}')
        _aliases[key] = canonical


def _pattern(alias):
    # ASCII boundaries allow Thai-adjacent abbreviations but never lab in collaborate.
    escaped = re.escape(alias).replace(r'\ ', r'\s*')
    return (r'(?<![a-z0-9])' if alias[0].isascii() and alias[0].isalnum() else '') + escaped + (
        r'(?![a-z0-9])' if alias[-1].isascii() else '')


_pattern_re = re.compile('|'.join(_pattern(a) for a in sorted(_aliases, key=len, reverse=True)), re.I)


def _canonical(match):
    value = clean_question(match.group()).casefold()
    return _aliases.get(value) or next(v for k, v in _aliases.items() if k.replace(' ', '') == value.replace(' ', ''))


@lru_cache(maxsize=2048)
def normalize_query(text):
    return _pattern_re.sub(_canonical, clean_question(text).casefold())


@lru_cache(maxsize=2048)
def expand_query(text):
    """Retain original intent, append just the canonical form, never all aliases."""
    original = clean_question(text)
    normalized = normalize_query(original)
    return original if original.casefold() == normalized else f'{original} {normalized}'


def matched_topics(text):
    return [(len(m.group()), _canonical(m)) for m in _pattern_re.finditer(clean_question(text).casefold())]
