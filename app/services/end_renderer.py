from app.domain.mappings import END_REASON_MAP


class EndRenderer:
    def __init__(self, llm):
        self.llm = llm

    async def render(self, messages, state):
        vars = self._end_vars(state)
        try:
            return await self.llm.debate_aware_end(
                messages=messages, prompt_vars=vars, temperature=0.2, max_tokens=80
            )
        except AttributeError:
            return await self.llm.debate_aware(
                messages=messages, state=state, temperature=0.2, max_tokens=80
            )

    def _end_vars(self, state):
        reason_label = state.last_judge_reason_label or 'unspecified_reason'
        end_reason = (
            END_REASON_MAP.get(reason_label)
            or state.end_reason
            or reason_label.replace('_', ' ')
        )
        return {
            'LANGUAGE': (state.lang or 'en').lower(),
            'TOPIC': state.topic,
            'DEBATE_STATUS': 'ENDED',
            'END_REASON': end_reason,
            'JUDGE_REASON_LABEL': reason_label,
            'JUDGE_CONFIDENCE': f'{state.last_judge_confidence:.2f}',
        }
