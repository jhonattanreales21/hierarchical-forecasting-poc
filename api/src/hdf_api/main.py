from fastapi import FastAPI

from hdf_api.routers import forecast

app = FastAPI(
    title="Demand Forecast POC — API",
    description="Serves precomputed forecast outputs produced by the Kedro pipelines.",
    version="0.1.0",
)

app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
