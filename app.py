import streamlit as st
import numpy as np
import joblib

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path(__file__).with_name("final_random_forest_model.pkl")
    return joblib.load(model_path)

model = load_model()

def main():
    st.header("Insurance Price Prediction", divider="gray")
    
    # Input fields
    age = st.number_input("Age", min_value=18, max_value=66, value=30)
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    bp_problems = st.selectbox("Blood Pressure Problems", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    transplants = st.selectbox("Any Transplants", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    chronic_diseases = st.selectbox("Any Chronic Diseases", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    height = st.number_input("Height in cm", min_value=145.0, max_value=188.0, value=170.0)
    weight = st.number_input("Weight in kg", min_value=51.0, max_value=132.0, value=70.0)
    allergies = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cancer_history = st.selectbox("History Of Cancer In Family", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    num_surgeries = st.number_input("Number Of Major Surgeries", min_value=0, max_value=3, value=0)

    if st.button("Predict Price", type="primary"):
        input_data = np.array([[age, diabetes, bp_problems, transplants, chronic_diseases, 
                              height, weight, allergies, cancer_history, num_surgeries]])
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Premium Price: ${prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()
