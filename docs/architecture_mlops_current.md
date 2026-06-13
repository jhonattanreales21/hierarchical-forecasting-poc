# Hierarchical Demand Forecasting PoC — Arquitectura MLOps (Estado Actual)

**Proyecto:** Forecasting Jerárquico Temporal de Demanda  
**Autores:** Jhonattan Reales · Andres Cano  
**Universidad:** Universidad Icesi — Maestría en Inteligencia Artificial Aplicada  
**Documento:** Descripción técnica de la arquitectura para sustentación

---

## 1. Visión General del Sistema

El proyecto implementa un sistema de pronóstico de demanda orientado a producción, estructurado en cinco capas funcionales que abarcan desde la ingestión de datos crudos hasta la presentación interactiva de resultados y la asistencia inteligente basada en recuperación de información. Su eje central es la capa de pronóstico mensual, concebida como el componente de mayor valor académico y de negocio.

La arquitectura sigue el principio **local-first, cloud-ready**: opera completamente en entornos locales con Docker, pero mantiene la estructura necesaria para despliegue en AWS (EC2, S3 opcional). El sistema está construido sobre un **monorepo de cuatro paquetes Python** gestionado con `uv` como workspace:

| Paquete | Directorio | Rol |
|--------|-----------|-----|
| `hdf_pipelines` | `pipelines/` | Proyecto Kedro — toda la lógica de datos y ML |
| `hdf_shared` | `shared/` | Librería interna — métricas, loaders, esquemas, RAG |
| `hdf_app` | `app/` | Interfaz Streamlit para el usuario de negocio |
| `hdf_api` | `api/` | Capa de servicio FastAPI |

---

## 2. Flujo de Datos: Ocho Capas del Catálogo

Todos los datasets están registrados en `catalog.yml` y ningún nodo escribe rutas de archivo de forma directa. El flujo atraviesa ocho capas ordenadas por nivel de transformación:

```
01_raw         →  02_intermediate  →  03_primary  →  04_feature
                                                           ↓
08_reporting   ←  07_model_output  ←  06_models   ←  05_model_input
```

| Capa | Propósito | Ejemplos de datasets |
|------|-----------|---------------------|
| **01_raw** | Archivos CSV de entrada | `raw_daily_demand`, `raw_exogenous_variables` |
| **02_intermediate** | Datos limpios y validados | `demand_cleaned`, `exogenous_cleaned` |
| **03_primary** | Demanda agregada por granularidad | `demand_monthly`, `demand_weekly`, `exogenous_monthly` |
| **04_feature** | Conjuntos de features construidos | `monthly_calendar_features`, `monthly_exogenous_features` |
| **05_model_input** | Splits listos para entrenamiento y ventanas futuras | `monthly_catboost_full_train`, `monthly_future_3m/6m/12m` |
| **06_models** | Artefactos de modelos, metadatos de tuning, campeones | `champion_monthly_model`, `champion_registry` |
| **07_model_output** | Pronósticos generados | `monthly_forecast_3m`, `monthly_forecast_latest` |
| **08_reporting** | Reportes de evaluación y explicabilidad | `monthly_family_champion_importance`, `monthly_catboost_shap_values` |

---

## 3. Pipelines Kedro: Arquitectura de Orquestación

El proyecto registra **11 pipelines** en `pipeline_registry.py`. El pipeline principal de producción (`__default__` / `monthly_forecast_e2e`) compone los siguientes stages en secuencia:

```
data_ingestion
    → feature_engineering_monthly
        → model_input_preparation
            → train_monthly (prophet + sarimax + catboost en paralelo)
                → monthly_model_selection
                    → forecast_inference
```

Los pipelines individuales por familia (`prophet_monthly_e2e`, `sarimax_monthly_e2e`, `catboost_monthly_e2e`) permiten ejecuciones aisladas para diagnóstico o comparación. Un pipeline legado `prophet_sarimax_comparison` se conserva por compatibilidad.

### 3.1 Pipeline: Data Ingestion

Responsabilidad: cargar, limpiar, anonimizar y agregar los datos crudos a las tres granularidades del sistema.

**Nodos principales:**
- `mask_raw_demand` — Anonimiza nombres de SKU (`sku1`, `sku2`, ...) y aplica un factor de escala configurable sobre la demanda
- `load_and_clean_demand` — Parsea fechas en formato M/D/YYYY, valida columnas obligatorias, elimina duplicados
- `load_and_clean_exogenous` — Normaliza fechas YYYY-MM, elimina espacios, agrega clave canónica de mes
- `build_demand_monthly` — Usa la demanda mensual reportada en la fuente como valor autoritativo (no suma de diarios), valida inconsistencias
- `build_demand_weekly` — Agrega demanda diaria a semanas ISO (inicio lunes)

**Contrato de entrada:**
- `raw_daily_demand.csv` — Columnas: SKU, Year, Month, Date, Monthly Demand, Daily Demand
- `raw_exogenous_variables.csv` — Columnas: Date, `pfizer_limited`, `surgifoam_limited`, `rebate_target`, `expected_market_share`

### 3.2 Pipeline: Feature Engineering (Mensual)

Responsabilidad: construir el conjunto de 19 features que alimentan los tres modelos candidatos.

**Tres grupos de features:**

**a) Features de calendario** (`build_monthly_calendar_features`)  
Generadas deterministamente a partir de la fecha. País configurado: Colombia (CO), máscara laboral: Lun–Vie.
- Días hábiles del mes (excluyendo festivos)
- Conteo de martes y jueves totales y hábiles
- Flags binarios: `has_5_working_tuesdays`, `has_5_working_thursdays`
- Conteo de festivos que caen en martes/jueves

**b) Features exógenas con rezagos** (`build_monthly_exogenous_features`)  
A partir de las cuatro columnas base: `pfizer_limited`, `surgifoam_limited`, `rebate_target`, `expected_market_share`.
- Rezagos de 1 y 2 meses para `pfizer_limited` y `surgifoam_limited`
- `market_share_stress` = `expected_market_share` ^ 1.5 (transformación no lineal)
- `market_share_uplift` = max(0, `expected_market_share` − 0.50)
- `expected_market_share` directo excluido como regresor (solo sus formas derivadas son activas)

**c) Dataset unificado** (`build_monthly_prophet_features`)  
Left-join entre demanda, calendario y exógenas. Una fila por (SKU, mes), ordenada cronológicamente.

**Total de features activos: 19**

### 3.3 Pipeline: Model Input Preparation

Responsabilidad: generar ventanas de rolling-origin y adaptar los datos al contrato de cada familia de modelos.

Este pipeline implementa el corazón metodológico del sistema: la **validación temporal con ventana expansiva (rolling-origin)**. A partir del dataset mensual completo, genera ciclos de entrenamiento con origen progresivo usando la librería compartida `shared/rolling_origin.py`.

**Parámetros globales de rolling-origin** (`parameters.yml`):

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `forecast_horizon` | 3 | Meses a predecir en cada ciclo |
| `n_cycles` | 5 | Número de ciclos de backtest |
| `rolling_window` | expanding | Ventana expansiva (toda la historia hasta el origen) |
| `step_months` | 1 | El origen avanza 1 mes por ciclo |
| `min_train_periods` | 24 | Mínimo de meses de entrenamiento en el primer ciclo |

**Ciclos generados (ejemplo con serie de 36 meses):**

```
Ciclo 1: Train [m1 – m28]  →  Test [m29, m30, m31]
Ciclo 2: Train [m1 – m29]  →  Test [m30, m31, m32]
Ciclo 3: Train [m1 – m30]  →  Test [m31, m32, m33]
Ciclo 4: Train [m1 – m31]  →  Test [m32, m33, m34]
Ciclo 5: Train [m1 – m32]  →  Test [m33, m34, m35]  ← último ciclo predice los meses más recientes
```

**Adaptadores por familia:**
- **Prophet:** Renombra `month_start_date` → `ds`, `monthly_demand` → `y`; genera frames de futuro (3m/6m/12m) con features exógenas proyectadas
- **SARIMAX:** Extrae columna objetivo y matriz de regresores exógenos por separado
- **CatBoost:** Formulación de target desplazado (una fila por horizonte h: features en origen t, target en t+h)

### 3.4 Pipeline: Train Monthly (Tres Familias)

Responsabilidad: búsqueda bayesiana de hiperparámetros y backtest rolling-origin para cada familia de modelos.

Las tres familias siguen el mismo protocolo unificado de entrenamiento:

1. **Optimización Bayesiana con Optuna (TPE — Tree-structured Parzen Estimator)**
2. **Backtest rolling-origin** por cada trial: el modelo se reentrena en cada uno de los 5 ciclos y predice los 3 meses siguientes
3. **Función objetivo:** minimizar WMAPE_M3 (WMAPE del horizonte 3 meses, promediado entre ciclos)
4. **Top-10 pre-campeones:** los 10 mejores candidatos por familia se reentrenan sobre el histórico completo para la etapa de selección

#### Familia Prophet

| Parámetro | Configuración |
|-----------|--------------|
| Trials máximos | 50 |
| Poda (Optuna pruning) | Habilitada desde trial 10, ciclo 2 |
| Espacio de búsqueda | `changepoint_prior_scale` [0.01–0.5], `seasonality_prior_scale` [1–10], `seasonality_mode` {additive/multiplicative} |
| Parámetros fijos | `yearly_seasonality: false`, `interval_width: 0.8` |
| Regresores | 19 features activos (modo aditivo) |

#### Familia SARIMAX

| Parámetro | Configuración |
|-----------|--------------|
| Trials máximos | 40 |
| Espacio de búsqueda | Órdenes (p,d,q) ∈ [0–3]², (P,D,Q,s=12) ∈ [0–2]² |
| Diagnóstico adicional | Test de Ljung-Box sobre residuos (lags=10, umbral p=0.05) |
| Variables exógenas | Sí (mismas 19 features) |

El test de Ljung-Box filtra candidatos con autocorrelación residual. En la selección, se priorizan candidatos que pasan el test; si ninguno pasa, se usa el conjunto completo como fallback.

#### Familia CatBoost

| Parámetro | Configuración |
|-----------|--------------|
| Trials máximos | 100 |
| Estrategia multi-horizonte | Directa: 3 modelos independientes (uno por h ∈ {1, 2, 3}) |
| Sin recursión | Las predicciones de h=1 no alimentan h=2; no hay propagación de error |
| Horizonte máximo | 3 meses (6m y 12m retornan vacío) |
| Espacio de búsqueda | `depth` [3–8], `learning_rate` [0.005–0.5], `iterations` [100–1000], `l2_leaf_reg` [1–10] |

### 3.5 Pipeline: Model Selection

Responsabilidad: elegir el campeón de producción mediante selección en dos niveles.

**Nivel 1 — Campeón de familia** (`select_monthly_family_champions`)  
Por cada familia activa (Prophet, SARIMAX, CatBoost):
1. Filtro SARIMAX: candidatos que pasan Ljung-Box primero; fallback al conjunto completo si ninguno pasa
2. Ranking por métrica primaria + desempate

**Nivel 2 — Campeón de producción** (`select_monthly_production_champion`)  
Entre los tres campeones de familia, se elige el mejor:

| Criterio | Orden |
|----------|-------|
| Métrica primaria | WMAPE_M3 (minimizar) |
| Desempate 1 | WMAPE_M2 |
| Desempate 2 | WMAPE_M1 |
| Desempate 3 | MASE |
| Desempate 4 | \|Bias\| absoluto |

**Artefactos de salida del campeón:**
- `champion_registry.json` — Registro centralizado: familia, `champion_id`, métrica de selección, valor
- `champion_monthly_model` (pickle) — Modelo reentrenado sobre todo el histórico
- `champion_monthly_metadata.json` — Metadatos completos de provenance

**Explicabilidad del campeón** (`explain_monthly_family_champion`):
- **CatBoost:** SHAP TreeSHAP (importancia global top-15 + valores SHAP por fila)
- **Prophet:** Importancia por contribución de componentes (regresores nativos Prophet)
- **SARIMAX:** Importancia de coeficientes del modelo ARIMA ajustado

### 3.6 Pipeline: Forecast Inference

Responsabilidad: generar las tablas de pronóstico finales a partir del campeón elegido.

El nodo `generate_monthly_champion_forecasts` implementa un **despacho dirigido por metadatos**: lee el campo `model_family` del `champion_monthly_metadata.json` y redirige la predicción al adaptador correspondiente (Prophet, SARIMAX, CatBoost). Este diseño permite agregar familias nuevas sin modificar el nodo principal.

**Esquema canónico de salida:**

```
date | forecast | forecast_lower | forecast_upper | model_family | granularity
horizon | horizon_label | forecast_generated_at | champion_id
selection_metric | selection_metric_value | sku | has_prediction_interval
interval_method | run_id | source_dataset
```

**Horizontes generados:** 3m, 6m, 12m (+ `monthly_forecast_latest` como copia del horizonte por defecto)

---

## 4. Lógica de Validación Temporal (Rolling-Origin)

La implementación de rolling-origin reside en `shared/src/shared/rolling_origin.py` como componente reutilizable. Es el núcleo metodológico del proyecto y garantiza la ausencia de data leakage temporal.

**Principios del protocolo:**

1. **Ventana expansiva:** cada ciclo usa toda la historia disponible hasta el origen, no solo una ventana de tamaño fijo
2. **Sin solapamiento train-test:** los meses de test nunca forman parte del entrenamiento del mismo ciclo
3. **Métricas por horizonte:** WMAPE_M1, WMAPE_M2, WMAPE_M3 permiten diagnosticar la degradación del pronóstico a medida que aumenta el horizonte
4. **Pooling de métricas:** WMAPE y BIAS se calculan agrupando numeradores y denominadores a través de todos los ciclos (no promediando WMAPEs individuales), lo que da mayor peso a ciclos con mayor volumen

**Métricas implementadas** (`shared/metrics.py`):

| Métrica | Descripción | Uso principal |
|---------|-------------|---------------|
| WMAPE | Weighted Absolute Percentage Error | Métrica de negocio; selección de campeón |
| WMAPE_Mh | WMAPE por horizonte h | Diagnóstico de degradación por horizonte |
| MASE | Mean Absolute Scaled Error | Comparación frente a naive estacional (s=12) |
| BIAS | Error promedio ponderado | Detección de sobre/sub-estimación sistemática |
| RMSE | Root Mean Squared Error | Penalización de errores grandes |

---

## 5. Capa de Aplicación

### 5.1 Streamlit (hdf_app)

La aplicación Streamlit opera como interfaz para el usuario de negocio y como trigger de los flujos de datos. Está organizada en cinco páginas:

| Página | Funcionalidad |
|--------|--------------|
| **01 — Data Upload** | Carga de demanda (CSV), variables exógenas y documentos de conocimiento (PDF, DOCX, MD, TXT). Validación de esquemas en tiempo real. |
| **02 — Descriptive Analysis** | Visualización de la serie histórica de demanda, variables exógenas, estadísticas descriptivas y descomposición temporal. |
| **03 — Monthly Forecast** | Tablas y gráficos de pronóstico para horizontes 3m, 6m y 12m. Intervalos de predicción al 80%. Selector de horizonte. |
| **04 — Evaluation Report** | Métricas rolling-origin por familia de modelos, justificación de selección de campeón, importancia de features (SHAP para CatBoost). |
| **05 — Business Assistant** | Asistente de Q&A basado en RAG sobre pronósticos, reportes y documentos cargados. |

**Principio de diseño:** Streamlit actúa como capa de interfaz y trigger. Toda la lógica de validación de datos, entrenamiento, selección de modelos e inferencia permanece en pipelines Kedro o en `hdf_shared`.

### 5.2 FastAPI (hdf_api)

La capa de API está scaffoldeada con tres endpoints definidos para servicio batch y on-demand:

| Endpoint | Descripción | Estado |
|----------|-------------|--------|
| `GET /health` | Health check del servicio | Implementado |
| `GET /forecast/latest` | Pronóstico más reciente por granularidad | Stub (501) |
| `GET /forecast/champion` | Metadata del campeón de producción | Stub (501) |

---

## 6. Capa de IA Generativa y RAG

El asistente de negocio (página 05 de Streamlit) implementa un flujo de **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre el pronóstico y el comportamiento histórico de la demanda.

**Componentes técnicos** (`shared/src/shared/`):

| Componente | Tecnología | Función |
|------------|------------|---------|
| Indexación de documentos | FAISS + MiniLM (`all-MiniLM-L6-v2`) | Índice vectorial local sobre documentos cargados |
| Chunking | 900 palabras, solapamiento 120 | Segmentación de documentos PDF/DOCX/MD/TXT |
| Recuperación | Top-5 chunks por consulta | Contexto más relevante para cada pregunta |
| LLM | Claude API | Generación de respuestas ejecutivas acotadas |
| Filtrado de scope | `is_in_scope()` | Rechaza preguntas fuera del dominio de pronóstico |
| Detección de período | `detect_period()` | Extrae año/mes de lenguaje natural para recuperar contexto histórico |

**Fuentes de conocimiento permitidas:**
- Tablas de pronóstico de `07_model_output/`
- Reportes de evaluación de `08_reporting/`
- Metadatos de modelos y registro de campeones de `06_models/`
- Documentación del proyecto en `pipelines/docs/` y `docs/`
- Logs de eventos de negocio cargados por el analista

**Restricciones de diseño del asistente:**
- Las respuestas están fundamentadas en recuperación: no se generan afirmaciones sin soporte documental
- No modifica ni sobreescribe outputs de modelos, métricas de evaluación ni metadatos de campeones
- Si el contexto recuperado es insuficiente, responde explícitamente que las fuentes disponibles no son suficientes para responder con confianza

---

## 7. Infraestructura y Reproducibilidad

### 7.1 Experiment Tracking (MLflow)

Los experimentos de tuning (Optuna) registran automáticamente hiperparámetros y métricas de cada trial. El seguimiento incluye: parámetros de configuración del trial, métricas rolling-origin (WMAPE_M3 objetivo + métricas por horizonte), estado del trial y timestamps. Los artefactos de los campeones (modelos, metadatos, configs) se persisten en `06_models/champions/`.

### 7.2 Configuración (OmegaConfigLoader)

El proyecto usa `OmegaConfigLoader` de Kedro con interpolación de variables entre archivos:

| Archivo | Ámbito |
|---------|--------|
| `conf/base/catalog.yml` | Todos los datasets del proyecto |
| `conf/base/parameters.yml` | Parámetros globales (horizonte, ciclos, semilla) |
| `conf/base/parameters/feature_engineering.yml` | Features de calendario y exógenas |
| `conf/base/parameters/train_monthly.yml` | Hiperparámetros de búsqueda por familia |
| `conf/base/parameters/model_selection.yml` | Métricas de selección, desempate, familias activas |
| `conf/local/` | Overrides locales (gitignoreado, nunca committeado) |

### 7.3 Docker

La infraestructura de contenedores en `docker/` soporta ejecución reproducible del proyecto completo, incluyendo los pipelines Kedro, la aplicación Streamlit y la API FastAPI.

---

## 8. Comparación con la Arquitectura Anterior

La imagen presentada en la versión anterior del diagrama definía cinco bloques (Business Intelligence, Data Engineering, Forecasting Factory, Explainable AI Layer, Deployment & Governance) con un ciclo de mejora continua (WMAPE Monitoring, Bias Tracking, Scenario Drift Detection, Retraining Trigger). A continuación se describe cómo cada bloque se relaciona con la arquitectura actual:

| Bloque (diagrama anterior) | Estado en arquitectura actual |
|---------------------------|-------------------------------|
| **Business Intelligence** — Planner Inputs, Historical Demand, Market Events, Exogenous Variables | **Implementado.** Ingesta de demanda diaria + exógenas. El repositorio de conocimiento de negocio es la capa RAG con FAISS + documentos del analista. |
| **Data Engineering** — Feature Store, Data Validation, Feature Engineering, Dataset Versioning | **Implementado.** Feature engineering en 3 grupos (calendario, exógenas, rezagos). Validación en nodos de ingesta. Versionado a través del catálogo Kedro (parquet nombrado por capa). |
| **Forecasting Factory** — Prophet, CatBoost, SARIMAX, Model Registry, Champion Selection, Forecast Generation | **Implementado.** Los tres modelos entrenan con Optuna + rolling-origin. El registro de campeones (`champion_registry.json`) es el Model Registry. Selección en dos niveles (family → production champion). Inferencia con despacho por metadatos. |
| **Explainable AI Layer** — Market History RAG, LLM Layer, Forecast Explanations, Driver Attribution, Business Context Retrieval | **Parcialmente implementado.** RAG funcional (FAISS + MiniLM + Claude API). Explicabilidad de features implementada (SHAP para CatBoost, componentes para Prophet, coeficientes para SARIMAX). Driver Attribution formal no implementado. |
| **Deployment & Governance** — Streamlit, Monitoring Dashboard, Bias Tracking, Retraining Pipeline, S&OP Approval, Governed Release | **Parcialmente implementado.** Streamlit funcional (5 páginas). FastAPI scaffoldeada (endpoints stub). Monitoring dashboard, retraining automático y governed release son trabajo futuro. |
| **Continuous Improvement Loop** — WMAPE Monitoring, Bias Tracking, Scenario Drift Detection | **No implementado (trabajo futuro).** El registro de campeones y las métricas de evaluación están disponibles como base para un monitor de drift, pero el loop automático no está activo. |

---

## 9. Decisiones de Diseño Clave

**1. Rolling-origin sin test hold-out separado**  
Los campeones se seleccionan exclusivamente sobre métricas de rolling-origin (ventana expansiva sobre toda la historia). No se reserva un hold-out test estático porque en contextos de negocio con series cortas esto reduciría el entrenamiento disponible de forma significativa. El ciclo más reciente predice los meses más recientes de la historia, actuando como proxy de test.

**2. Selección en dos niveles (family → production)**  
Desacopla la optimización dentro de cada familia de la comparación entre familias. Permite aplicar reglas específicas por familia (como el filtro Ljung-Box para SARIMAX) sin afectar el protocolo global de selección.

**3. Estrategia directa multi-horizonte para CatBoost**  
Tres modelos independientes por horizonte (h=1, h=2, h=3) evitan la propagación de error de los enfoques recursivos. La limitación al horizonte de 3 meses es intencional: CatBoost opera como modelo tabular puro y no tiene representación probabilística nativa para horizontes largos.

**4. Despacho dirigido por metadatos en inferencia**  
Un único nodo de inferencia lee el campo `model_family` del `champion_monthly_metadata.json` y redirige a tres adaptadores especializados. Este patrón permite cambiar de campeón entre familias sin modificar el código del pipeline de inferencia.

**5. RAG como capa de interpretabilidad, no de decisión**  
El asistente GenAI está diseñado explícitamente para no modificar ni reinterpretar los outputs de los modelos. Toda respuesta debe estar fundamentada en documentos recuperados del índice; si el contexto es insuficiente, el sistema lo declara.

---

## 10. Resumen Ejecutivo de la Arquitectura

El sistema implementa un **pipeline MLOps completo** para pronóstico de demanda mensual con las siguientes características centrales:

- **Orquestación reproducible** mediante Kedro (11 pipelines, 8 capas de datos, catálogo centralizado)
- **Validación temporal rigurosa** con rolling-origin de 5 ciclos y ventana expansiva, sin leakage
- **Búsqueda bayesiana de hiperparámetros** (Optuna TPE) para tres familias: Prophet, SARIMAX y CatBoost
- **Protocolo de selección en dos niveles**: campeón de familia → campeón de producción, con métricas de desempate explícitas
- **Explicabilidad integrada**: SHAP (CatBoost), importancia de componentes (Prophet), coeficientes (SARIMAX)
- **Inferencia dirigida por metadatos**: el campeón puede ser de cualquier familia sin cambios en el pipeline de inferencia
- **Asistente de negocio RAG**: FAISS + MiniLM + Claude API, con scope-filtering y contexto histórico dinámico
- **Interfaz de usuario completa**: Streamlit con 5 páginas (carga de datos, análisis, pronóstico, evaluación, asistente)
- **Capa de servicio**: FastAPI scaffoldeada para servicio batch y on-demand

La arquitectura es modular, extensible a granularidades semanales (pipeline semanal scaffoldeado) y reconciliación jerárquica temporal (pipeline `reconciliation` en el registro, configuración `mint_shrink` disponible).
