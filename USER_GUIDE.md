# User Guide — Demand Forecasting App

A friendly, step-by-step guide to get the project running on your computer, generate
the forecasts, open the app, and understand what each screen does.

---

## 1. What this project does

It forecasts the monthly demand of a critical product (SKU). It trains several
forecasting models, automatically picks the best one (the "champion"), and shows the
results in a web app where you can explore history, see the forecast, check how good the
models are, and ask questions in plain language.

---

## 2. Quick install

You only do this once.

### Step 1 — Install the two requirements

- **Python 3.12** → https://www.python.org/downloads/
- **uv** (the tool that installs everything else). Open a terminal and paste:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> On Windows, use PowerShell: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Step 2 — Get the project and install it

```bash
# Download the project
git clone https://github.com/jhonattanreales21/hierarchical-forecasting-poc.git
cd hierarchical-forecasting-poc

# Install everything (this can take a few minutes the first time)
uv sync --all-packages
```

That's it. The project is installed.

---

## 3. Quick run — generate the forecasts

Before the app can show anything, the models need to run at least once. There's a
shortcut for this:

```bash
make run
```

This single command runs the whole pipeline end to end: it reads the data, builds
features, trains the models (SARIMAX, Prophet, CatBoost), picks the champion, and
generates the forecast. When it finishes, the results are saved and ready for the app.

> ⏳ This takes a few minutes. It's normal to see a lot of text scrolling — that's the
> models being trained.

**Want to check it worked?** You can list what was produced:

If you ever want to re-run just one part instead of everything, these shortcuts exist
(optional, most people just use `make run`):

```bash
make ingest          # only load and clean the data
make train-monthly   # only train the monthly models
make model-selection # only pick the champion model
make infer           # only generate the forecast
```

---

## 4. Open the app (run it locally)

With the forecasts generated, launch the web app:

```bash
make app
```

Then open your browser at **http://localhost:8501**

> The same thing can be done with the longer command
> `uv run --package hdf_app streamlit run app/app.py` — but `make app` is easier.

To stop the app, go back to the terminal and press `Ctrl + C`.

---

## 5. How to navigate the app

When the app opens you'll see the **Overview** page with cards linking to each section.
There's also a **top menu bar** to jump between pages at any time. Here's what each
page is for.

### 🏠 Overview
The landing page. It just introduces the app and gives you quick links to every section.
Start here and move left to right.

### 📤 Data Upload
Where you bring in **new data** to refresh the analysis.

- **Demand + Exogenous CSV** — upload your demand history and the external variables
  (both files are required before the "Submit data" button turns on). The app checks the
  files (rows, columns, date range, granularity) and saves them.
- **Assistant Knowledge Document** — upload a business-history document (PDF, Word,
  Markdown, or TXT). This feeds the Business Assistant so it can explain *why* things
  happened.
- You'll see a summary of each file after it's saved, plus a list of your latest uploads.

> 💡 Good to know: uploading here **saves and validates** the files but does **not**
> retrain the models by itself. To actually refresh the forecasts with new data, run
> `make run` again in the terminal.

### 📊 Descriptive Analysis
Explore the **history before any forecasting**. Here you can look at past demand at
different time levels (daily, weekly, monthly) and see how the external variables moved
over time. Use this to get a feel for trends, seasonality, and spikes.

### 📈 Monthly Forecast
The **main page** — this is the core output of the project.

- **Horizon selector** — choose how far ahead to look (e.g. 3, 6, or 12 months).
- **KPI cards** — headline numbers about the forecast and the champion model.
- **Forecast chart** — past demand plus the predicted future. If the champion model
  produces uncertainty ranges, you'll see a shaded band around the line.
- **Forecast table + download** — the exact predicted numbers, which you can download.
- **Champion model details** — which model won and its key settings.
- **Data status** — a quick note on what data is available.

> If this page says "Champion Model Not Found", it just means the models haven't run
> yet — go back to step 3 and run `make run`.

### 📋 Evaluation Report
The **"why should I trust this?"** page. It explains how the winning model was chosen
and how all the candidates performed.

- The champion selection rationale and a quick business-quality flag.
- A comparison of the best model from each family (SARIMAX, Prophet, CatBoost).
- Which factors mattered most for the predictions (explainability).
- A full table of every candidate's error metrics.
- Validation notes about how testing was done.

### 🤖 Business Assistant
Ask questions in **plain language** about the forecast and history.

- **Build the knowledge index** (right panel) — if you uploaded a document on the Data
  Upload page, click to index it so the assistant can use it.
- **Ask a question** (left panel) — for example *"Why does the forecast change in June
  2026?"*. The assistant answers using the forecast, the history, and your documents,
  and you can expand "Evidence sent to assistant" to see exactly what it used.
- **Forecast visuals** — switch between historical demand and the 12-month outlook.

> The assistant only answers questions related to the forecast and demand. It won't
> invent business explanations — if it doesn't have enough information, it will say so.

---

## 6. Typical first-time flow

1. Install once → `uv sync --all-packages` (section 2)
2. Generate forecasts → `make run` (section 3)
3. Open the app → `make app`, then go to http://localhost:8501 (section 4)
4. Explore: **Descriptive Analysis** → **Monthly Forecast** → **Evaluation Report**
5. (Optional) Upload a business document and chat with the **Business Assistant**

---

## 7. Common questions

**The app shows "Champion Model Not Found" / "No evaluation data".**
The models haven't run yet. Run `make run` in the terminal, then refresh the app.

**I uploaded new data but the forecast didn't change.**
Uploading only saves the files. Run `make run` again to retrain with the new data.

**How do I stop the app?**
In the terminal where it's running, press `Ctrl + C`.

**Is there an API too?**
Yes (optional, for developers). Start it with:
`uv run --package hdf_api uvicorn hdf_api.main:app --reload --port 8000`
and open http://localhost:8000/docs

---

Need more technical detail? See the [README](README.md) and the docs in
[docs/](docs/).
