import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the Excel file
file_name = "finalresult.csv"
df = pd.read_csv(file_name)

# Select features and target variable
X = df.drop({'meter_reading'}, axis=1)  # Drop the 'meter_reading' column, keeping the other features
y = df["meter_reading"]  # Choose the 'electricity' column as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
rf_regressor.fit(X_train, y_train)

# Make predictions using the model
y_pred = rf_regressor.predict(X_test)

# Calculate the model's Mean Squared Error and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error of the model:", mse)
print("R² score of the model:", r2)


