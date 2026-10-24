
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "marks": [20, 25, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

# Separate input and output
X = df[["study_hours"]]
y = df["marks"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the ML pipeline
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
predictions = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Training completed!")
print("Predicted marks:", predictions.round(2))
print("Mean Absolute Error:", round(mae, 2))

# Save the trained model
joblib.dump(model_pipeline, "student_marks_pipeline.pkl")

print("Model saved successfully!")
