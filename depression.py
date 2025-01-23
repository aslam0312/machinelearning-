
import streamlit as st
import pandas as pd
import joblib

# Load the logistic regression model
lr_model = joblib.load('model/LinearRegressionModel.pkl')

# Data dictionary for the dataset
data_dictionary = {
    "Age": "Age of the individual",
    "Academic_Pressure": "Level of academic pressure on a scale of 0-5",
    "CGPA": "Cumulative Grade Point Average",
    "Study_Satisfaction": "Satisfaction with studies on a scale of 0-5",
    "Have_you_ever_had_suicidal_thoughts_": "History of suicidal thoughts (Yes/No)",
    "Work_Study_Hours": "Number of hours spent working/studying per day",
    "Financial_Stress": "Level of financial stress on a scale of 0-5",
    "Family_History_of_Mental_Illness": "Family history of mental illness (Yes/No)",
    "Sleep_Duration": "Average sleep duration (5-6_hours, 7-8_hours, etc.)",
    "Dietary_Habits": "Dietary habits (Healthy/Moderate/Unhealthy)"
}

# Streamlit app
st.set_page_config(page_title="Depression Prediction App", layout="centered", initial_sidebar_state="auto")
st.title("Depression Prediction App")

# Display data dictionary
st.header("Data Dictionary")
data_dict_df = pd.DataFrame(list(data_dictionary.items()), columns=["Feature", "Description"])
st.table(data_dict_df)

# Introduction section
st.header("Understanding Depression")
st.write(
    "Depression is a common mental disorder that negatively affects how a person feels, thinks, and acts. "
    "It can lead to various emotional and physical problems, reducing one's ability to function at work or home. "
    "This app aims to predict depression risks based on user inputs, enabling early intervention and support."
)

# User input form
st.header("User Input Form")

# Custom styles for input widgets
st.markdown(
    """
    <style>
    .stTextInput, .stSlider, .stSelectbox, .stRadio {
        font-size: 18px !important;
    }
    label {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Collect inputs
Age = st.number_input('Age', min_value=0, max_value=100)
Academic_Pressure = st.slider('Academic Pressure', min_value=0, max_value=5)
CGPA = st.number_input('CGPA', min_value=0.0, max_value=15.0, value=3.31, step=0.01)
Study_Satisfaction = st.slider('Study Satisfaction', min_value=0, max_value=5)
Sleep_Duration = st.selectbox('Sleep Duration', [
    '5-6_hours', '7-8_hours', 'Less_than_5_hours', 'More_than_8_hours'
])
Dietary_Habits = st.selectbox('Dietary Habits', ['Healthy', 'Moderate', 'Unhealthy'])
Have_you_ever_had_suicidal_thoughts_ = st.radio('Have you ever had suicidal thoughts?', ('Yes', 'No'))
Work_Study_Hours = st.slider('Work/Study Hours', min_value=0, max_value=13)
Financial_Stress = st.slider('Financial Stress', min_value=0, max_value=5)
Family_History_of_Mental_Illness = st.radio('Family History of Mental Illness', ('Yes', 'No'))

# Create input DataFrame
user_data = pd.DataFrame({
    'Age': [Age],
    'Academic_Pressure': [Academic_Pressure],
    'CGPA': [CGPA],
    'Study_Satisfaction': [Study_Satisfaction],
    'Have_you_ever_had_suicidal_thoughts_': [1 if Have_you_ever_had_suicidal_thoughts_ == 'Yes' else 2],
    'Work_Study_Hours': [Work_Study_Hours],
    'Financial_Stress': [Financial_Stress],
    'Family_History_of_Mental_Illness': [1 if Family_History_of_Mental_Illness == 'Yes' else 2]
})

# Adding one-hot encoded features
sleep_columns = [f'Sleep_Duration_{duration}' for duration in ['5-6_hours', '7-8_hours', 'Less_than_5_hours', 'More_than_8_hours']]
diet_columns = [f'Dietary_Habits_{diet}' for diet in ['Healthy', 'Moderate', 'Unhealthy']]

for col in sleep_columns + diet_columns:
    user_data[col] = 1 if col == f'Sleep_Duration_{Sleep_Duration}' or col == f'Dietary_Habits_{Dietary_Habits}' else 0

# Ensure all features match the model
all_features = list(lr_model.feature_names_in_)
for col in all_features:
    if col not in user_data:
        user_data[col] = 0

user_data = user_data[all_features]  # Ensure feature order matches

# Make predictions
st.subheader("Prediction")
if st.button('Predict'):
    try:
        # Predict using the logistic regression model
        lr_prediction = lr_model.predict(user_data)[0]
      
        # Display results
        if lr_prediction == 1:  #Assuming 0.5 as the threshold for high risk
            st.image("image2.png", caption="Prediction: High Risk")
            st.subheader("You are depressed. please go through the given url and help  yourself")
            st.page_link('https://www.nhs.uk/mental-health/self-help/tips-and-support/cope-with-depression/',label='https://www.nhs.uk/mental-health/self-help/tips-and-support/cope-with-depression/')
        else:
            st.image("image1.png", caption="Prediction: Low Risk")
            st.subheader("You are not depressed. Keep up the good work!")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
