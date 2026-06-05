"""GrassVision custom exceptions and error handlers."""
from fastapi import Request
from fastapi.responses import JSONResponse


class GrassVisionError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ConfigError(GrassVisionError):
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class ProviderError(GrassVisionError):
    def __init__(self, message: str, provider: str = "", status_code: int = 502):
        self.provider = provider
        super().__init__(message, status_code=status_code)


class ImageError(GrassVisionError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ModelNotFoundError(GrassVisionError):
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not found or disabled", status_code=404)


class VisionAnalysisError(GrassVisionError):
    def __init__(self, message: str):
        super().__init__(message, status_code=502)


async def grassvision_exception_handler(request: Request, exc: GrassVisionError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": type(exc).__name__}},
    )
