import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("birthRate_data.csv")
df.drop(['Code'], axis=1, inplace = True)
df.rename(columns={'Fertility rate - Sex: all - Age: all - Variant: estimates': 'Fertility', 'Entity':'Country'}, inplace = True)

df2 = df.groupby('Year')['Fertility'].mean().reset_index()

X = df2['Year'].values.reshape(-1, 1)
y = df2['Fertility'].values

model = LinearRegression()
model.fit(X, y)

last_year = df2['Year'].max()
future_years = np.array(range(last_year + 1, last_year + 21)).reshape(-1, 1)
future_predictions = model.predict(future_years)

plt.figure(figsize=(12, 6))
plt.plot(df2['Year'], df2['Fertility'], marker='o', label='Historical Data')
plt.plot(future_years, future_predictions, 'r--', label='Predictions')

critical_rate = 2.1
plt.axhline(y=critical_rate, color='r', linestyle='-', alpha=0.5, label='Replacement Level (2.1)')
plt.fill_between([df2['Year'].min(), future_years.max()], 0, critical_rate, 
                 color='red', alpha=0.1, label='Extinction Risk Zone')

plt.title('Global Fertility Rate: Historical Data and Future Predictions')
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.grid(True)
plt.legend()

slope = model.coef_[0]
intercept = model.intercept_
plt.text(0.02, 0.95, f'Prediction equation: y = {slope:.4f}x + {intercept:.4f}',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.show()

print("\nPrediction Details:")
print(f"Current trend: {slope:.4f} fertility rate units per year")
print("\nPredicted fertility rates for next 5 years:")
for year, pred in zip(future_years[:5], future_predictions[:5]):
    print(f"{year[0]}: {pred:.2f}")


