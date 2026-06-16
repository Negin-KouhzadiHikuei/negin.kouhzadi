import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print('Connecting to global servers and retrieving real energy consumption data ')

url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
df_global = pd.read_csv(url)

#print("Success! The world's huge data entered Python's memory without downloading it to the hard drive.")
#print(f"Dimensions of the loaded data: {df_global.shape[0]} rows and {df_global.shape[1]} columns! (A real, gigantic data)")


df_armenia = df_global[df_global['country'] == 'Armenia'].copy()
columns_needed = ['year', 'population', 'gdp', 'energy_per_capita']
df_clean = df_armenia[columns_needed].dropna()

print(f"\n📊 Surgical multi-variable data for Armenia: {df_clean.shape[0]} clean rows found.")
print(df_clean.head())


x = df_clean[['year', 'population', 'gdp']]
y = df_clean['energy_per_capita']


ai_model = LinearRegression()
ai_model.fit(x, y)

print("\n🎯 Boom! Multi-variable AI model trained successfully on population, year, and GDP relations!")

predictions = ai_model.predict(x)
plt.figure(figsize=(10, 5))
plt.plot(df_clean['year'], y, label='Actual Data', color='darkcyan', marker='o')
plt.plot(df_clean['year'], predictions, label='AI Prediction', color='red', linestyle='--', marker='s')
plt.title('Multi-Variable AI Model: Armenia Energy Consumption Over Time')
plt.xlabel('Year')
plt.ylabel('Energy Consumption per Capita')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

