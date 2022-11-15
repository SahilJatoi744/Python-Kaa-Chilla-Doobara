import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Making container for the sidebar
header   = st.container()
dataset  = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Kashti Dashboard')
    st.text('This project is about the "Titanic" dataset.')
    
with dataset:
    st.header('Dataset')
    # Importing the dataset
    df = sns.load_dataset('titanic')
    # Removing the NaN values
    df = df.dropna()
    
    st.subheader('Dataset Preview')
    st.write(df.head(3))
    st.subheader('Dataset Description')
    st.write(df.describe())
    
    # Plotting the graphs 
    st.subheader('Survived Passerngers')
    st.bar_chart(df.survived.value_counts())
        
    st.subheader('Fare Distribution')
    st.bar_chart(df.sex.value_counts())
    
    st.subheader('Class Distribution')
    st.bar_chart(df['class'].value_counts())
    
    st.subheader('Age Distribution - 25 Random Passengers')
    st.bar_chart(df.age.sample(25))    
        
with model_training:
    st.header('Kashti Passengers Status - Model Training')
    st.text('This is the model training.')
    
    # Making the columns
    input, display = st.columns(2)
    
    # Input column
    max_depth = input.slider('Age', min_value=0, max_value=100, value=15, step=2)
    
# n_estimators
n_estimators = input.selectbox('How many trees in the forest?', options=[50, 100, 200, 300, 'No Limit'])

# Defining the X and y
X = df['age'].values.reshape(-1, 1)
y = df['fare'].values.reshape(-1, 1)

# Machine Learning Model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

# Applying condition for the model
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

# Fitting the model
model.fit(X,y)
pred = model.predict(y)

# Displaying metrics
display.subheader('Mean Absolute Error:')
display.write(mean_absolute_error(y, pred))
display.subheader('Mean Squared Error:')
display.write(mean_squared_error(y, pred))
display.subheader('R Squared Score:')
display.write(r2_score(y, pred)) 
