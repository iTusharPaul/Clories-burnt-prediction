# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:56:19 2024

@author: KIIT
"""
 
import numpy as np  #for creating input array
import pickle    #for loading saved model
import streamlit as st #for creating user interface


#loading the saved model
loaded_model = pickle.load(open('loaded_model.sav','rb'))


def caloriePrediction (input_data):
    input_data_as_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_array.reshape(1,-1)
     # this will reshape the data into a single row with the required number of columns.
     #This is done so that the algo treats this data as a single data point.
   
    prediction = loaded_model.predict(input_data_reshaped)
    # we are using loaded model from the file for prediction
    return prediction[0]


def main():
    st.title('Calories Burnt Predictor')

    #getting user inputs using text fields
    Gender = st.text_input('Enter Gender:Male(0)/Female(1)')
    Age = st.text_input('Enter your Age (in yrs)')
    Height = st.text_input('Enter your height (in cms)')
    Weight = st.text_input('Enter weight (in kgs)')
    Duration = st.text_input('Enter duration of workout (in mins)')
    Heart_Rate = st.text_input('Enter heart rate (in BPM)')
    Body_Temp = st.text_input('Enter Body Temperature (in celcius)') 
    
    
    #making prediction using defined function
    calories = 0
    
    #if button is pressed. model will be used for getting prediction
    if(st.button('Predict My Calories Burnt')):
        input_data = [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
        input_data_reshaped = np.array(input_data, dtype=np.float64).reshape(1, -1)
        calories = caloriePrediction (input_data_reshaped)
        #if successfully runned success meassgae will be the output:
        st.success(f"You have successfully Burnt {calories} calories")
    
    
    
if __name__ == '__main__':
    main()
#the above two lines is used so that this file cannot be imported and run using some other file. 
#only if this file is run as a stand alone file. this will run the main function.
    
    