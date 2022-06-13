# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:52:03 2022

@author: ANURAG
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model

loaded_model = pickle.load(open("C:/Users/ANURAG/DATASCIENCEPRACTICE/Project20_Deploying_Diabetes_Prediction_Model_using_Streamlit/trained_model.sav", "rb"))

# Creating a function for Prediction

def diabetes_prediction(input_data):
    
    

    #changing the input data to a numpy array

    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    # Standarize nahi kar rahe hai

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0])==0:
        return "The Person is Non Diabetic"
    else:
        return "The Person is Diabetic"
        
        
        
def main():
    
    
    #Giving a title 
    
    st.title("Diabetes Prediction Web App")
    
    # Getting the input data from the user
    
        
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")
    
    
    #Code for prediction
    
    diagnosis = ''
    
    
    #Creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    