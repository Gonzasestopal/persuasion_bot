# app/api/errors.py
from fastapi import FastAPI, Request
from starlette import status as st
from starlette.responses import JSONResponse

from app.domain import errors as de


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(de.InvalidStartMessage)
    async def _422_start(_: Request, exc: de.InvalidStartMessage):
        return JSONResponse(
            status_code=st.HTTP_422_UNPROCESSABLE_ENTITY,
            content={'detail': exc.message},
        )

    @app.exception_handler(de.InvalidContinuationMessage)
    async def _422_cont(_: Request, exc: de.InvalidContinuationMessage):
        return JSONResponse(
            status_code=st.HTTP_422_UNPROCESSABLE_ENTITY,
            content={'detail': exc.message},
        )

    @app.exception_handler(de.InvalidTopic)
    async def _422_topic(_: Request, exc: de.InvalidContinuationMessage):
        return JSONResponse(
            status_code=st.HTTP_422_UNPROCESSABLE_ENTITY,
            content={'detail': exc.message},
        )

    @app.exception_handler(de.ConversationNotFound)
    async def _404_not_found(_: Request, exc: de.ConversationNotFound):
        return JSONResponse(
            status_code=st.HTTP_404_NOT_FOUND, content={'detail': exc.message}
        )

    @app.exception_handler(de.ConversationExpired)
    async def _404_expired(_: Request, exc: de.ConversationExpired):
        # Keep your existing contract/text if tests expect it:
        return JSONResponse(
            status_code=st.HTTP_404_NOT_FOUND, content={'detail': exc.message}
        )

    @app.exception_handler(de.LLMTimeout)
    async def _503_timeout(_: Request, exc: de.LLMTimeout):
        return JSONResponse(
            status_code=st.HTTP_503_SERVICE_UNAVAILABLE, content={'detail': exc.message}
        )

    @app.exception_handler(de.LLMServiceError)
    async def _502_llm(_: Request, exc: de.LLMServiceError):
        # or 503 if you prefer; just be consistent with tests
        return JSONResponse(
            status_code=st.HTTP_502_BAD_GATEWAY, content={'detail': exc.message}
        )

    @app.exception_handler(de.ConfigError)
    async def _500_config_error(_, exc: de.ConfigError):
        return JSONResponse(
            status_code=st.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'detail': exc.message},
        )
