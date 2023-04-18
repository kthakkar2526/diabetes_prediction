# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:29:21 2023

@author: KT
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('D:/streamlit/trained_model1.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_np_array = np.asarray(input_data)
input_data_reshaped = input_data_as_np_array.reshape(1,-1)
#std_data = scaler.transform(input_data_reshaped)
print(input_data_reshaped)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("person is non-diabetic")
else:
    print("person is diabetic") 