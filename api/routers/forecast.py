from fastapi import APIRouter, HTTPException

from shared.schemas import BacktestResult, ForecastRecord, TemporalGranularity

router = APIRouter()


@router.get("/latest", response_model=list[ForecastRecord])
async def get_latest_forecast(
    granularity: TemporalGranularity = TemporalGranularity.MONTHLY,
    reconciled: bool = True,
):
    """Return the latest precomputed forecast for the requested granularity.

    Reads from data/07_model_output/ once the Kedro pipeline has been run.
    - granularity: MONTHLY (default, primary layer) or WEEKLY (operational anchor).
      DAILY is currently disabled at the parameter level.
    - reconciled: if True, return the reconciled forecast; if False, return
      the raw model output. Both variants are produced by the Kedro pipeline.

    On-demand inference is a future extension.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — run the Kedro pipeline first.",
    )


@router.get("/backtest", response_model=list[BacktestResult])
async def get_backtest_results(
    granularity: TemporalGranularity = TemporalGranularity.MONTHLY,
):
    """Return evaluation results (MAPE, RMSE, MASE) for all trained models.

    Reads from data/08_reporting/evaluation_report.parquet once the
    evaluation pipeline has been run.

    Monthly results are the primary reporting output and the main
    success criterion for this POC.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — run the evaluation pipeline first.",
    )


@router.get("/champion", response_model=dict)
async def get_champion_registry(
    granularity: TemporalGranularity = TemporalGranularity.MONTHLY,
):
    """Return the champion model registry for the requested granularity.

    Reads from data/06_models/champions/champion_registry.json once
    model selection has been run.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — run the model_selection pipeline first.",
    )
