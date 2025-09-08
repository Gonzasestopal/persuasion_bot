from typing import Dict

END_REASON_MAP: Dict[str, str] = {
    'strict_thesis_contradiction': 'TYour arguments directly contradicted the thesis and prevailed. Congratulations, you won this debate.',
    'strong_contradiction_evidence': 'Compelling evidence strongly contradicted the defended thesis.',
    'supports_defended_stance': 'The user’s argument actually supported the defended thesis.',
    'policy_threshold_reached': 'Enough of your points were accepted. Well done, you convinced me.',
    'max_turns_reached': 'Debate ended after reaching the maximum number of turns.',
    'unspecified_reason': 'The debate concluded per policy.',
}
