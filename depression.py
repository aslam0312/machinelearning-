# # 

# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained model
# model = joblib.load('model.pkl')

# # Streamlit UI
# st.write("""
# # Depression Prediction App
# """)
# st.sidebar.header('User Input Parameters')

# # Define the user input function
# def user_input_features():
#     # Gender as radio buttons
#     Gender = st.sidebar.radio('Gender', ('Male', 'Female'))
#     Gender_value = 1 if Gender == 'Male' else 2

#     # Age as manual input
#     Age = st.sidebar.number_input('Age', min_value=0, max_value=100)

#     # Suicidal thoughts as radio buttons
#     Have_you_ever_had_suicidal_thoughts_ = st.sidebar.radio('Have you ever had suicidal thoughts?', ('Yes', 'No'))
#     Have_you_ever_had_suicidal_thoughts_value = 1 if Have_you_ever_had_suicidal_thoughts_ == 'Yes' else 2

#     # Family history of mental illness as radio buttons
#     Family_History_of_Mental_Illness = st.sidebar.radio('Family History of Mental Illness', ('Yes', 'No'))
#     Family_History_of_Mental_Illness_value = 1 if Family_History_of_Mental_Illness == 'Yes' else 2

#     # City as dropdown
#     City = st.sidebar.selectbox('City', [
#         'Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi', 'Faridabad', 'Ghaziabad',
#         'Hyderabad', 'Indore', 'Jaipur', 'Kalyan', 'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana',
#         'Meerut', 'Mumbai', 'Nagpur', 'Nashik', 'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
#         'Thane', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Visakhapatnam'
#     ])
#     City_value = f'City_{City}'

#     # Degree as dropdown
#     Degree = st.sidebar.selectbox('Degree', ['Class_12', 'Bachelor', 'Master'])
#     Degree_value = f'Degree_{Degree}'

#     # Sleep duration as dropdown
#     Sleep_Duration = st.sidebar.selectbox('Sleep Duration', [
#         '5-6_hours', '7-8_hours', 'Less_than_5_hours', 'More_than_8_hours'
#     ])
#     Sleep_Duration_value = f'Sleep_Duration_{Sleep_Duration}'

#     # Dietary habits as dropdown
#     Dietary_Habits = st.sidebar.selectbox('Dietary Habits', ['Healthy', 'Moderate', 'Unhealthy'])
#     Dietary_Habits_value = f'Dietary_Habits_{Dietary_Habits}'

#     # Manual and slider input for numerical features
#     CGPA = st.sidebar.slider('CGPA', min_value=0.0, max_value=15.0, value=3.31, step=0.01)
#     Academic_Pressure = st.sidebar.slider('Academic Pressure', min_value=0, max_value=15, value=6)
#     Study_Satisfaction = st.sidebar.slider('Study Satisfaction', min_value=0, max_value=15, value=5)
#     Financial_Stress = st.sidebar.slider('Financial Stress', min_value=0, max_value=15, value=5)
#     Work_Study_Hours = st.sidebar.slider('Work/Study Hours', min_value=0, max_value=15, value=13)

#     # Compile all inputs into a dictionary
#     data = {
#         'Gender': Gender_value,
#         'Age': Age,
#         'Academic_Pressure': Academic_Pressure,
#         'CGPA': CGPA,
#         'Study_Satisfaction': Study_Satisfaction,
#         'Have_you_ever_had_suicidal_thoughts_': Have_you_ever_had_suicidal_thoughts_value,
#         'Work_Study_Hours': Work_Study_Hours,
#         'Financial_Stress': Financial_Stress,
#         'Family_History_of_Mental_Illness': Family_History_of_Mental_Illness_value,
#         City_value: 1,  # Encode the selected City
#         Degree_value: 1,  # Encode the selected Degree
#         Sleep_Duration_value: 1,  # Encode the selected sleep duration
#         Dietary_Habits_value: 1,  # Encode the selected dietary habit
#     }

#     return data

# # Generate the input DataFrame
# data = user_input_features()

# # Ensure all required features from the model are present
# required_features = model.feature_names_in_  # Extract required feature names from the model
# df = pd.DataFrame(0, index=[0], columns=required_features)  # Create a DataFrame with all required features, initialized to 0
# df.update(pd.DataFrame(data, index=[0]))  # Update with user input data

# # Display user input
# st.subheader('User Input Parameters')
# st.write(df)

# # Make predictions
# if st.button('Predict'):
#     try:
#         prediction = model.predict(df)
#         st.subheader('Prediction')
#         st.write(prediction)
#     except ValueError as e:
#         st.error(f"Error during prediction: {e}")


import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Streamlit UI
st.write("""
# Depression Prediction App
""")
st.sidebar.header('User Input Parameters')

# Define the user input function
def user_input_features():
    # Gender as radio buttons
    Gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    Gender_value = 1 if Gender == 'Male' else 2

    # Age as manual input
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100)

    # Suicidal thoughts as radio buttons
    Have_you_ever_had_suicidal_thoughts_ = st.sidebar.radio('Have you ever had suicidal thoughts?', ('Yes', 'No'))
    Have_you_ever_had_suicidal_thoughts_value = 1 if Have_you_ever_had_suicidal_thoughts_ == 'Yes' else 2

    # Family history of mental illness as radio buttons
    Family_History_of_Mental_Illness = st.sidebar.radio('Family History of Mental Illness', ('Yes', 'No'))
    Family_History_of_Mental_Illness_value = 1 if Family_History_of_Mental_Illness == 'Yes' else 2

    # City as dropdown
    City = st.sidebar.selectbox('City', [
        'Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi', 'Faridabad', 'Ghaziabad',
        'Hyderabad', 'Indore', 'Jaipur', 'Kalyan', 'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana',
        'Meerut', 'Mumbai', 'Nagpur', 'Nashik', 'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
        'Thane', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Visakhapatnam'
    ])
    City_value = f'City_{City}'

    # Degree as dropdown
    Degree = st.sidebar.selectbox('Degree', ['Class_12', 'Bachelor', 'Master'])
    Degree_value = f'Degree_{Degree}'

    # Sleep duration as dropdown
    Sleep_Duration = st.sidebar.selectbox('Sleep Duration', [
        '5-6_hours', '7-8_hours', 'Less_than_5_hours', 'More_than_8_hours'
    ])
    Sleep_Duration_value = f'Sleep_Duration_{Sleep_Duration}'

    # Dietary habits as dropdown
    Dietary_Habits = st.sidebar.selectbox('Dietary Habits', ['Healthy', 'Moderate', 'Unhealthy'])
    Dietary_Habits_value = f'Dietary_Habits_{Dietary_Habits}'

    # Manual and slider input for numerical features
    CGPA = st.sidebar.slider('CGPA', min_value=0.0, max_value=15.0, value=3.31, step=0.01)
    Academic_Pressure = st.sidebar.slider('Academic Pressure', min_value=0, max_value=5)
    Study_Satisfaction = st.sidebar.slider('Study Satisfaction', min_value=0, max_value=5)
    Financial_Stress = st.sidebar.slider('Financial Stress', min_value=0, max_value=5)
    Work_Study_Hours = st.sidebar.slider('Work/Study Hours', min_value=0, max_value=13)

    # Compile all inputs into a dictionary
    data = {
        'Gender': Gender_value,
        'Age': Age,
        'Academic_Pressure': Academic_Pressure,
        'CGPA': CGPA,
        'Study_Satisfaction': Study_Satisfaction,
        'Have_you_ever_had_suicidal_thoughts_': Have_you_ever_had_suicidal_thoughts_value,
        'Work_Study_Hours': Work_Study_Hours,
        'Financial_Stress': Financial_Stress,
        'Family_History_of_Mental_Illness': Family_History_of_Mental_Illness_value,
        City_value: 1,  # Encode the selected City
        Degree_value: 1,  # Encode the selected Degree
        Sleep_Duration_value: 1,  # Encode the selected sleep duration
        Dietary_Habits_value: 1,  # Encode the selected dietary habit
    }

    return data

# Generate the input DataFrame
user_data = user_input_features()

# Ensure all required features from the model are present
required_features = model.feature_names_in_  # Extract required feature names from the model
df = pd.DataFrame(0, index=[0], columns=required_features)  # Create a DataFrame with all required features, initialized to 0
df.update(pd.DataFrame(user_data, index=[0]))  # Update with user input data

# Prepare a clean display of only non-zero user inputs
display_data = {key: value for key, value in user_data.items() if value != 0}

# Display user input
st.subheader('User Input Parameters')
st.write(pd.DataFrame(display_data, index=[0]))

# Make predictions
if st.button('Predict'):
    try:
        prediction = model.predict(df)
        st.subheader('Prediction')
        st.write(prediction)
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
