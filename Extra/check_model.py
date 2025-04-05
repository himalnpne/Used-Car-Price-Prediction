import joblib

# Load the model
model = joblib.load('car_price_model.pkl')

# Check if the model is fitted
print(model)