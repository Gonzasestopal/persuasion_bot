# app/shared/text_utils.py
import re
from typing import Dict, List

# regex compilados una vez, multi-idioma
WORD_RX = re.compile(r'[^\W\d_]+', flags=re.UNICODE)
SENT_SPLIT_RX = re.compile(r'(?<=[.!?¿\?¡!])\s+')
IS_QUESTION_RX = re.compile(r'[¿\?]\s*$')
END_MARKERS_RX = re.compile(
    r'(match concluded\.?|debate concluded|debate is over)',
    flags=re.IGNORECASE,
)


def trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + '…'


def round3(d: Dict[str, float]) -> Dict[str, float]:
    return {
        k: round(float(d.get(k, 0.0)), 3)
        for k in ('entailment', 'neutral', 'contradiction')
    }


def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def sanitize_end_markers(text: str) -> str:
    return normalize_spaces(END_MARKERS_RX.sub('', text))


def drop_questions(text: str) -> str:
    sents = [s.strip() for s in re.split(SENT_SPLIT_RX, text) if s.strip()]
    sents = [s for s in sents if not IS_QUESTION_RX.search(s)]
    out = ' '.join(sents) if sents else text
    return re.sub(r'\.\.+$', '.', out).strip()


def word_count(text: str) -> int:
    return len(WORD_RX.findall(text))


SPANISH_Q_WORDS = (
    '¿',
    'como',
    'cómo',
    'que',
    'qué',
    'por que',
    'por qué',
    'cuando',
    'cuándo',
    'donde',
    'dónde',
    'cual',
    'cuál',
    'cuales',
    'cuáles',
    'quien',
    'quién',
    'quienes',
    'quiénes',
)


def looks_like_question(s: str) -> bool:
    s2 = (s or '').strip().lower()
    if not s2:
        return False
    return s2.startswith('¿') or any(s2.startswith(w + ' ') for w in SPANISH_Q_WORDS)


def _ends_with_strong_punct(s: str) -> bool:
    return s.endswith(('.', '!', '?', '…'))


def strip_trailing_fragment(parts: List[str]) -> List[str]:
    # if the original text doesn't end with strong punctuation, drop last fragment (often a cut question)
    return parts[:-1] if parts and not _ends_with_strong_punct(parts[-1]) else parts


_STOP_EN = {
    'the',
    'a',
    'an',
    'of',
    'to',
    'and',
    'or',
    'in',
    'on',
    'for',
    'with',
    'by',
    'is',
    'are',
    'was',
    'were',
    'be',
    'being',
    'been',
    'it',
    'this',
    'that',
    'these',
    'those',
    'as',
    'at',
    'from',
    'not',
    'no',
    'yes',
    'but',
    'if',
    'then',
    'so',
    'than',
    'because',
    'i',
    'you',
    'he',
    'she',
    'we',
    'they',
    'them',
    'me',
    'my',
    'your',
    'our',
    'their',
    'his',
    'her',
    'its',
    'what',
    'which',
    'who',
    'whom',
    'how',
    'why',
    'when',
    'where',
    'there',
    'here',
}
_STOP_ES = {
    'el',
    'la',
    'los',
    'las',
    'un',
    'una',
    'unos',
    'unas',
    'de',
    'del',
    'al',
    'a',
    'y',
    'o',
    'en',
    'con',
    'por',
    'para',
    'según',
    'sin',
    'sobre',
    'es',
    'son',
    'fue',
    'eran',
    'ser',
    'siendo',
    'sido',
    'esto',
    'esa',
    'ese',
    'esas',
    'esos',
    'estos',
    'esas',
    'como',
    'que',
    'qué',
    'quien',
    'quién',
    'cuando',
    'cuándo',
    'donde',
    'dónde',
    'porqué',
    'porque',
    'no',
    'sí',
    'pero',
    'si',
    'yo',
    'tú',
    'vos',
    'usted',
    'ustedes',
    'él',
    'ella',
    'nosotros',
    'nosotras',
    'ellos',
    'ellas',
    'mi',
    'mis',
    'tu',
    'tus',
    'su',
    'sus',
    'nuestro',
    'nuestra',
    'nuestros',
    'nuestras',
    'aquí',
    'allí',
    'ahí',
}
STOP_ALL = _STOP_EN | _STOP_ES


ACK_PREFIXES = (
    'thanks',
    'thank you',
    'i appreciate',
    'good point',
    'fair point',
    'i see',
    'understand',
)

# Exclude stance/meta banners from claim extraction
STANCE_BANNERS = (
    'i will gladly take the pro stance',
    'i will gladly take the con stance',
    'i will defend the proposition as stated',
    'defenderé la proposición tal como está',
    'defenderei a proposição como está',
    'tomaré el lado pro',
    'tomaré el lado con',
    'tomarei o lado pro',
    'tomarei o lado con',
)
