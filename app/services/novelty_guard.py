import re
import string
from typing import List

from app.utils.text import STOP_ALL, trunc


class NoveltyGuard:
    def __init__(self, scoring):
        self.scoring = scoring

    def compute(self, convo, latest_user_idx):
        latest = convo[latest_user_idx].get('content', '')
        prev = [
            m.get('content', '')
            for m in convo[:latest_user_idx]
            if m.get('role') == 'user' and m.get('content')
        ]
        return self._novelty_score(latest, prev)

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _novelty_score(self, current: str, previous_texts: List[str]) -> float:
        """
        Returns novelty in [0,1]. 1.0 = completely new, 0.0 = duplicated.
        Uses a max-similarity penalty against any prior user message.
        """
        if not current:
            return 0.0
        if not previous_texts:
            return 1.0

        cur_ngrams = self._char_ngrams(current, n=3)
        cur_tokens = set(self._tokenize_norm(current))

        max_sim = 0.0
        for prev in previous_texts:
            prev_ngrams = self._char_ngrams(prev, n=3)
            prev_tokens = set(self._tokenize_norm(prev))

            jacc = self._jaccard(cur_ngrams, prev_ngrams)
            tok_sim = (
                len(cur_tokens & prev_tokens) / max(len(cur_tokens), len(prev_tokens))
                if (cur_tokens or prev_tokens)
                else 1.0
            )
            sim = 0.65 * jacc + 0.35 * tok_sim
            if sim > max_sim:
                max_sim = sim

        novelty = 1.0 - max_sim
        return max(0.0, min(1.0, novelty))

    def _latest_user_novelty(self, convo: List[dict], latest_user_idx: int) -> float:
        """
        Compute novelty of the latest user message vs all *prior* user messages.
        """
        latest_txt = convo[latest_user_idx].get('content', '')
        prev_users = [
            m.get('content', '')
            for i, m in enumerate(convo[:latest_user_idx])
            if m.get('role') == 'user' and m.get('content')
        ]
        return self._novelty_score(latest_txt, prev_users)

    @staticmethod
    def _tokenize_norm(s: str) -> List[str]:
        # Alphanumeric tokens, lowercased, punctuation stripped
        if not s:
            return []
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation + '¿¡“”"…—–-'))
        tokens = re.findall(r'\b\w+\b', s)
        return [t for t in tokens if len(t) > 2 and t not in STOP_ALL]

    @staticmethod
    def _char_ngrams(s: str, n: int = 3) -> set:
        if not s:
            return set()
        s2 = s.lower()
        s2 = re.sub(r'\s+', ' ', s2).strip()
        s2 = s2.translate(
            str.maketrans('', '', ' \t\n\r' + string.punctuation + '¿¡“”"…—–-')
        )
        if len(s2) < n:
            return {s2} if s2 else set()
        return {s2[i : i + n] for i in range(len(s2) - n + 1)}
