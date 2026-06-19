import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================================
# 📊 PHASE 1: DATA INGESTION, CLEANING & ANONYMIZATION
# ==========================================================
print(" Ingesting live energy indicators from cloud storage...")
url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
df = pd.read_csv(url)

# Filter for the target region and select multi-variable features
df_region = df[df['country'] == 'Armenia']
features = ['year', 'population', 'gdp', 'energy_per_capita']
df_clean = df_region[features].dropna()

# Extract feature matrix (X) and target vector (y)
X = df_clean[['year', 'population', 'gdp']]
y = df_clean['energy_per_capita']


#  PHASE 2: MULTI-VARIABLE MODEL TRAINING & VISUALIZATION
# ==========================================================
print(" Constructing and training Multi-Variable Linear Regression engine...")
ai_model = LinearRegression()
ai_model.fit(X, y)

# Generate high-fidelity baseline predictions for target verification
predictions = ai_model.predict(X)

# Render the production analytics chart
plt.figure(figsize=(10, 5))
plt.plot(df_clean['year'], y, label='Actual Historical Data', color='darkcyan', marker='o')
plt.plot(df_clean['year'], predictions, label='AI Predictive Alignment', color='red', linestyle='--')
plt.title('Multi-Variable AI Model: Regional Energy Ingestion Analytics')
plt.xlabel('Timeline (Years)')
plt.ylabel('Energy Consumption per Capita')
plt.legend()
plt.grid(visible=True, linestyle=':')

# Automatically save the visualization for the secure GitHub asset pipeline
plt.savefig('multivariable_prediction.png', dpi=300)
print("📸 Analytics visualization successfully exported as 'multivariable_prediction.png'.")
plt.show()


#  PHASE 3: FUTURE SCENARIO INFERENCE (YEAR 2030)
# ==========================================================
print("\n Transitioning AI core into future forecasting matrix...")

future_year = 2030
estimated_population = 3100000.0  # Projected macro demographic growth
estimated_gdp = 2.8e+10           # Projected macro economic growth

# Structure the input as a 2D frame for formal inference
future_scenario = pd.DataFrame([[future_year, estimated_population, estimated_gdp]],
                               columns=['year', 'population', 'gdp'])

# Run the forecast model
predicted_energy_2030 = ai_model.predict(future_scenario)

print("=" * 70)
print(f" [AI Production Forecast 2030]:")
print(f"   Target Estimated Energy Consumption per Capita: {predicted_energy_2030[0]:.3f}")
print("=" * 70)