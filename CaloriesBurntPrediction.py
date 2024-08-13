# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle




#loading the saved model
loaded_model = pickle.load(open('C:/Users/KIIT/Desktop/Machine Learning Specialization/projects/Clories burnt prediction/loaded_model.sav','rb'))

input_data = (0,68,190.0,94.0,29.0,105.0,40.8)
input_data_as_array = np.asarray(input_data)

input_data_reshaped = input_data_as_array.reshape(1,-1)
 # this will reshape the data into a single row with the required number if columns.
 #This is done so that the algo treats this data as a single data point.
print(input_data_reshaped)


prediction = loaded_model.predict(input_data_reshaped) # we are using loaded model from the file for prediction
print(f"You have Successfully burnt {prediction} calories!!!")