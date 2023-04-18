# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:37:24 2023

@author: KT
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/streamlit/trained_model1.sav', 'rb'))

#creating a function fro prediction 

def DiabetesPrediction(input_data):
   
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    #std_data = scaler.transform(input_data_reshaped)
    print(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "person is non-diabetic"
    else:
        return "person is diabetic"
    
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure Level')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    #code for prediction
    diagnosis = '' 
    
    #creating a button
    if st.button('Diabetes test result'):
        diagnosis = DiabetesPrediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,  Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
        
        
        
        