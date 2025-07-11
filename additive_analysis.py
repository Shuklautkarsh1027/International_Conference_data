
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('additive_manufacturing_data.csv')

# 1. Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Additive Manufacturing')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 2. Scatter Plot: Porosity vs Temperature
fig1 = px.scatter(df, x='Temperature_C', y='Porosity_Percent', color='Infill_Percent',
                  title='Porosity vs Temperature Colored by Infill %')
fig1.write_html('porosity_vs_temperature.html')

# 3. Pairplot for full comparison
sns.pairplot(df)
plt.savefig('pairplot_all_parameters.png')
plt.close()

# 4. Outlier Detection (Z-score method)
z_scores = np.abs((df[['Porosity_Percent']] - df[['Porosity_Percent']].mean()) / df[['Porosity_Percent']].std())
df['Outlier'] = z_scores > 2

# 5. PCA for dimensionality reduction and feature impact
features = ['Layer_Height_mm', 'Temperature_C', 'Infill_Percent', 'Porosity_Percent']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
fig2 = px.scatter(df, x='PCA1', y='PCA2', color='Porosity_Percent', title='PCA Projection')
fig2.write_html('pca_projection.html')

# 6. Porosity Distribution Plot
sns.histplot(df['Porosity_Percent'], kde=True)
plt.title('Distribution of Porosity %')
plt.savefig('porosity_distribution.png')
plt.close()

# 7. Linear Regression: Predicting Porosity
model = LinearRegression()
model.fit(df[['Layer_Height_mm', 'Temperature_C', 'Infill_Percent']], df['Porosity_Percent'])
df['Predicted_Porosity'] = model.predict(df[['Layer_Height_mm', 'Temperature_C', 'Infill_Percent']])
plt.scatter(df['Porosity_Percent'], df['Predicted_Porosity'])
plt.xlabel('Actual Porosity')
plt.ylabel('Predicted Porosity')
plt.title('Regression Fit: Actual vs Predicted Porosity')
plt.savefig('regression_fit.png')
plt.close()

# 8. Infill Optimization Zone (55–70%)
optimal_infill = df[(df['Infill_Percent'] >= 55) & (df['Infill_Percent'] <= 70)]
optimal_infill.to_csv('optimal_infill_range.csv', index=False)

# 9. Boxplot by Infill % ranges
df['Infill_Range'] = pd.cut(df['Infill_Percent'], bins=[0, 50, 60, 70, 100])
sns.boxplot(x='Infill_Range', y='Porosity_Percent', data=df)
plt.title('Porosity vs Infill Ranges')
plt.savefig('porosity_infill_boxplot.png')
plt.close()

# 10. Save final processed data
df.to_csv('processed_am_data.csv', index=False)
