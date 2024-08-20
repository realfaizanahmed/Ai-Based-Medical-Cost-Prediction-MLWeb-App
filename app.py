import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

# Load the dataset
try:
    df = pd.read_csv('medical_cost.csv')
except FileNotFoundError:
    st.error("The file 'medical_cost.csv' was not found.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The file 'medical_cost.csv' is empty.")
    st.stop()
except pd.errors.ParserError:
    st.error("Error parsing the file 'medical_cost.csv'.")
    st.stop()

# Prepare the data
X = df.drop(columns=['charges'])
y = df['charges']

# Define preprocessing for numeric and categorical features
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Create transformers for preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define and train the Linear Regression model
lr_model = Ridge()
lr_model.fit(X_train, y_train)

# Build and train the Feedforward Neural Network model
fnn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Single output for regression
])

fnn_model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
fnn_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Combine predictions for the training and testing sets
train_preds_combined = np.column_stack((lr_model.predict(X_train), fnn_model.predict(X_train).flatten()))
test_preds_combined = np.column_stack((lr_model.predict(X_test), fnn_model.predict(X_test).flatten()))

# Train the meta-model (another Linear Regression)
meta_model = LinearRegression()
meta_model.fit(train_preds_combined, y_train)

# Evaluate the hybrid model
meta_test_preds = meta_model.predict(test_preds_combined)
mse = mean_squared_error(y_test, meta_test_preds)
mae = mean_absolute_error(y_test, meta_test_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, meta_test_preds)

# Streamlit App Title
st.title("Medical Insurance Cost Prediction App")
st.write("Use the sidebar to enter your details and click 'Predict' to see your medical insurance cost prediction:")

# Sidebar for User Input
st.sidebar.header("Enter Your Details")

# Collect all user inputs within a form
with st.sidebar.form("prediction_form"):
    age = st.number_input("Age", min_value=0, max_value=100, value=25, help="Enter your age")
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, help="Enter your Body Mass Index")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, help="Enter the number of children")
    sex = st.selectbox("Sex", options=['male', 'female'], help="Select your sex")
    smoker = st.selectbox("Smoker", options=['yes', 'no'], help="Do you smoke?")
    region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'], help="Select your region")

    # Add a submit button inside the form
    submitted = st.form_submit_button("Predict")

# Prediction Logic after form submission
if submitted:
    user_input = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }

    # Function to Predict Insurance Cost
    def predict_cost(user_input):
        user_df = pd.DataFrame([user_input], columns=X.columns)
        user_preprocessed = preprocessor.transform(user_df)
        lr_pred = lr_model.predict(user_preprocessed)
        fnn_pred = fnn_model.predict(user_preprocessed).flatten()
        combined_pred = np.column_stack((lr_pred, fnn_pred))
        final_pred = meta_model.predict(combined_pred)
        return final_pred[0]

    # Predict and Display the Result
    predicted_cost = predict_cost(user_input)
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;">
        <h3 style="color: #155724;">Predicted Insurance Cost: ${predicted_cost:.2f}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

    # Display Model Performance Metrics
    st.write("### Model Performance Metrics")
    st.write(f"- **Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"- **Mean Absolute Error (MAE)**: {mae:.2f}")
    st.write(f"- **Root Mean Squared Error (RMSE)**: {rmse:.2f}")
    st.write(f"- **R-squared (R²)**: {r2:.2f}")