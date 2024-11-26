import shap
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare the data
data = pd.read_excel('MockData.xlsx')
print('Read excel')
X = data[['UNRATE', 'T10YIEM']]
y = data['SP500']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Split training sets.')

# Define and train the model
model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=6, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"Root Mean Squared Error: {rmse:.2f}")

explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.plots.beeswarm(shap_values)