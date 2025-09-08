from typing import Dict

END_REASON_MAP: Dict[str, str] = {
    'strict_thesis_contradiction': 'The user directly contradicted the main thesis without reconciliation.',
    'strong_contradiction_evidence': 'Compelling evidence strongly contradicted the defended thesis.',
    'supports_defended_stance': 'The user’s argument actually supported the defended thesis.',
    'policy_threshold_reached': 'Debate ended by policy threshold.',
    'max_turns_reached': 'Debate ended after reaching the maximum number of turns.',
    'unspecified_reason': 'The debate concluded per policy.',
}
