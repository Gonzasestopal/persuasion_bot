# app/adapters/judge/openai_score_judge.py
import json
from typing import Optional

from openai import OpenAI

from app.domain.ports.scoring import ScoreFeatures, ScoreJudgePort, ScoreVerdict

_SYSTEM = """You are a strict meta-judge. You must ONLY analyze the numeric features provided.
Do not infer from missing text. Apply the following logic preferences:

- Primary signal: thesis_contradiction vs thesis_entailment against thresholds.
- If thesis is underdetermined, consider pair_contradiction and pair_confident.
- Concession = true when user opposes the bot's stance with strong, confident contradiction.
- Alignment:
  - 'OPPOSITE' if contradiction >> entailment (or strong pairwise contradiction),
  - 'SAME' if entailment >> contradiction,
  - 'UNKNOWN' otherwise.
Return STRICT JSON only, no prose, matching the schema.
"""

_JSON_SCHEMA = {
    'name': 'ScoreVerdict',
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'required': ['alignment', 'concession', 'reason', 'confidence'],
        'properties': {
            'alignment': {'enum': ['OPPOSITE', 'SAME', 'UNKNOWN']},
            'concession': {'type': 'boolean'},
            'reason': {'type': 'string'},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        },
    },
}


def _features_to_prompt(f: ScoreFeatures) -> str:
    # Keep keys stable and explicit for reproducibility
    return json.dumps(f, ensure_ascii=False, separators=(',', ':'))


class OpenAIScoreJudge(ScoreJudgePort):
    def __init__(
        self, api_key: str, model: str = 'gpt-4o-mini', temperature: float = 0.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def score(self, *, features: ScoreFeatures) -> Optional[ScoreVerdict]:
        user = _features_to_prompt(features)
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {'role': 'system', 'content': _SYSTEM},
                {'role': 'user', 'content': user},
            ],
            temperature=self.temperature,
            response_format={'type': 'json_schema', 'json_schema': _JSON_SCHEMA},
        )
        try:
            data = resp.output_text
            return json.loads(data)  # type: ignore
        except Exception:
            return None
