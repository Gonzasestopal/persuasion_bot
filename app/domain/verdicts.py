from app.domain.concession_policy import DebateState

AFTER_END_MESSAGE = {
    'en': 'The debate has already ended. Please start a new conversation if you want to debate another topic.',
    'es': 'El debate ya terminó. Por favor inicia una nueva conversación si quieres debatir otro tema.',
    'pt': 'O debate já terminou. Por favor, inicie uma nova conversa se quiser debater outro tema.',
    # ... add more as needed
}


def after_end_message(state: DebateState) -> str:
    return AFTER_END_MESSAGE.get(state.lang)
