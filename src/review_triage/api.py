"""API HTTP e interface web de demonstração."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import Settings, get_settings
from .io import load_csv, parse_csv
from .providers import LLMProvider, build_provider
from .schemas import ReviewIn, TriageRequest, TriageResponse
from .service import run_triage

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_PATH = DATA_DIR / "reviews_sample.csv"
DEMO_PATH = DATA_DIR / "demo_results.json"


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.provider = provider or build_provider(settings)
        yield
        await app.state.provider.aclose()

    app = FastAPI(
        title="Review Triage AI",
        description="Triagem de avaliações de clientes com LLMs.",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def _triage(reviews: list[ReviewIn], include_summary: bool) -> TriageResponse:
        if len(reviews) > settings.max_reviews_per_request:
            raise HTTPException(
                413, f"Máximo de {settings.max_reviews_per_request} avaliações por requisição"
            )
        return await run_triage(settings, app.state.provider, reviews, include_summary)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "provider": settings.llm_provider, "model": settings.model}

    @app.get("/sample", response_model=list[ReviewIn])
    async def sample() -> list[ReviewIn]:
        """Avaliações sintéticas de exemplo para a demo."""
        return load_csv(SAMPLE_PATH) if SAMPLE_PATH.exists() else []

    @app.get("/demo", response_model=TriageResponse)
    async def demo() -> TriageResponse:
        """Resultado pré-calculado com um LLM sobre as avaliações de exemplo."""
        if not DEMO_PATH.exists():
            raise HTTPException(404, "Resultado de demonstração não gerado")
        return TriageResponse.model_validate_json(DEMO_PATH.read_text(encoding="utf-8"))

    @app.post("/triage", response_model=TriageResponse)
    async def triage(request: TriageRequest) -> TriageResponse:
        return await _triage(request.reviews, request.include_summary)

    @app.post("/triage/csv", response_model=TriageResponse)
    async def triage_csv(file: UploadFile, include_summary: bool = True) -> TriageResponse:
        try:
            reviews = parse_csv((await file.read()).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(422, f"CSV inválido: {exc}") from exc
        if not reviews:
            raise HTTPException(422, "Nenhuma avaliação encontrada no CSV")
        return await _triage(reviews, include_summary)

    return app


app = create_app()
