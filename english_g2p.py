"""English words -> Russian phoneme symbols for the glados/TeraTTS model.

English g2p_en (ARPAbet) is mapped to the model's 62 RU phoneme symbols so the
Russian VITS model can pronounce English words in a Russian accent. No retrain.
"""
from __future__ import annotations

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


ARPA_VOWEL = {
    "AA": "a", "AE": "a", "AH": "a", "AO": "o",
    "EH": "e", "IH": "i", "IY": "i", "UH": "u", "UW": "u",
    "AW": "a", "AY": "a", "EY": "e", "OW": "o", "OY": "o", "ER": "e",
}
ARPA_DIPH_TAIL = {
    "AW": ["u0"], "AY": ["j"], "EY": ["j"], "OW": ["u0"], "OY": ["j"], "ER": ["r"],
}
ARPA_CONS = {
    "B": ["b"], "CH": ["ch"], "D": ["d"], "DH": ["z"], "F": ["f"],
    "G": ["g"], "HH": ["h"], "JH": ["d", "zh"], "K": ["k"], "L": ["l"],
    "M": ["m"], "N": ["n"], "NG": ["n"], "P": ["p"], "R": ["r"],
    "S": ["s"], "SH": ["sh"], "T": ["t"], "TH": ["s"], "V": ["v"],
    "W": ["v"], "Y": ["j"], "Z": ["z"], "ZH": ["zh"],
}


def arpa_to_ru_symbols(arpa_phones):
    """Convert ARPAbet phones (e.g. ['HH','AH0','L','OW1']) to model RU symbols."""
    out = []
    for p in arpa_phones:
        if not p or not p[0].isalpha():
            continue
        stress = ""
        base = p
        if p[-1].isdigit():
            base, stress = p[:-1], p[-1]
        if base in ARPA_VOWEL:
            ru_stress = "1" if stress in ("1", "2") else "0"
            out.append(ARPA_VOWEL[base] + ru_stress)
            out.extend(ARPA_DIPH_TAIL.get(base, []))
        elif base in ARPA_CONS:
            out.extend(ARPA_CONS[base])
    return out


def english_to_ru_symbols(text):
    """Full path: English text -> ARPAbet (g2p_en) -> RU phoneme symbols.

    Word separators (g2p_en emits ' ' between words) become a literal ' '
    symbol so the tokenizer inserts a word gap.
    """
    arpa = _get_g2p()(text)
    syms = []
    for tok in arpa:
        if tok == " ":
            syms.append(" ")
        else:
            syms.extend(arpa_to_ru_symbols([tok]))
    return syms


def _is_cyrillic(ch):
    return "\u0400" <= ch <= "\u04FF"


def _is_latin(ch):
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def split_by_script(text):
    """Split text into contiguous (run_text, 'ru'|'en') segments.

    Non-alphabetic chars attach to the current run. Adjacent same-lang runs are
    merged. A leading non-alpha run defaults to 'ru' (Russian-first server)."""
    if not text:
        return []
    segments = []
    cur_lang = "ru"
    cur = ""
    for ch in text:
        if _is_cyrillic(ch):
            if cur_lang != "ru" and cur:
                segments.append((cur, cur_lang)); cur = ""
            cur_lang = "ru"; cur += ch
        elif _is_latin(ch):
            if cur_lang != "en" and cur:
                segments.append((cur, cur_lang)); cur = ""
            cur_lang = "en"; cur += ch
        else:
            cur += ch
    if cur:
        segments.append((cur, cur_lang))
    merged = []
    for seg in segments:
        if merged and merged[-1][1] == seg[1]:
            merged[-1] = (merged[-1][0] + seg[0], seg[1])
        else:
            merged.append(seg)
    return merged
