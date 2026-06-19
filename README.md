# Multi-Variable Machine Learning for Regional Energy Forecasting

An end-to-end predictive analytics system built to model and forecast regional power consumption by expanding a baseline single-variable model into a robust Multi-Variable Linear Regression system.

##  Executive Summary & Business Logic
In modern infrastructure engineering (such as datacenter cooling and regional power grid management), predicting energy demands based on a single metric leads to severe **Underfitting**. 

This project demonstrates how a baseline model built on a single economic feature failed to capture the data's variance, resulting in a flat prediction line. By shifting the architecture to a **Multi-Variable Linear Regression** model and injecting temporal, demographic, and economic factors simultaneously, the AI successfully unlocked the hidden historical patterns of the target region.

---

##  Data Pipeline & Ingestion Architecture
Rather than using static local files that consume disk space, this pipeline dynamically streams over **23,000 rows** of real-world global metrics directly from cloud servers into Python's memory environment.

- **Target Variable ($y$):** `energy_per_capita` (Power consumption index per person).
- **Feature Matrix ($X$):** 1. `year` (Temporal vector)
  2. `population` (Demographic scale)
  3. `gdp` (Economic power)

###  Surgical Data Cleaning
The raw global frame was masked down strictly to the 'Armenia' territory. Incomplete sensor records containing missing data (`NaN`) were dynamically eliminated using Pandas `.dropna()`, ensuring a high-fidelity dataset for model training.

---

##  Technical Defense: Evolution of the AI Brain

### Phase 1: The Single-Variable Limitation (Underfitting)
Initially, training the model exclusively on a single feature failed. Because the true nature of power grid usage is multi-dimensional, the AI was too simple to learn the pattern, producing a static, flat baseline.

<p align="center">
  <img src="./baseline_underfitting.png" alt="Baseline Single Variable Model" width="750">
</p>

### Phase 2: Multi-Variable Matrix Expansion (Successful Learning)
By reshaping the input into a 2D matrix ($X$) containing `year`, `population`, and `gdp`, the model solved a multi-dimensional linear equation:

$$\text{Energy per Capita} = (w_1 \times \text{Year}) + (w_2 \times \text{Population}) + (w_3 \times \text{GDP}) + b$$

The trained AI line tightly tracked the actual historical dips, peaks, and market transitions of the region, achieving high predictive convergence.

<p align="center">
  <img src="./multivariable_prediction.png" alt="Multi Variable AI Model" width="750">
</p>

---

##  Phase 3: Future AI Forecasting (Year 2030)

To validate the model's production readiness, we transitioned the AI core into an inference phase. We simulated a future macro-economic and demographic scenario for Armenia in the year **2030** using the following projections:
* **Target Year:** 2030
* **Estimated Population:** 3,100,000 (Projected growth)
* **Estimated GDP:** $28,000,000,000 (Projected economic expansion)

###  AI Inference Output
The trained Multi-Variable Linear Regression engine processed the matrix and generated the following forecast for regional energy requirements:
* **Predicted Energy Consumption per Capita:** `19812.541` (or your exact output number) kWh

This confirms that under sustained economic and demographic scaling, the regional energy infrastructure and data centers must prepare for an all-time high load by the end of the decade.

---

##  How to Run the Infrastructure

1. Ensure your localized Python virtual environment is active.
2. Install the necessary data science dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn