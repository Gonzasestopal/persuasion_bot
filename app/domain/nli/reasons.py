# app/domain/nli/reasons.py
from enum import Enum


class JudgeReason(str, Enum):
    USER_DEFENDS_PRO_THESIS = 'user_defends_pro_thesis'
    USER_DEFENDS_CON_THESIS = 'user_defends_con_thesis'
    STRICT_THESIS_CONTRADICTION = 'strict_thesis_contradiction'
    AMBIGUOUS_EVIDENCE = 'ambiguous_evidence'
    OFF_TOPIC = 'off_topic'
    POLICY_TURN_LIMIT = 'policy_turn_limit'
    POSITIVE_JUDGEMENTS_REACHED = 'positive_judgements_reached'


ALLOWED_REASONS = {r.value for r in JudgeReason}


ALIASES = {
    'user_defending_same_stance': 'user_defends_pro_thesis',
    'same_stance': 'user_defends_pro_thesis',
    'opposite_stance': 'user_defends_con_thesis',
    'thesis_opposition_soft': 'strict_thesis_contradiction',
}
