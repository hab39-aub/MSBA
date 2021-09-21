import streamlit as st
import numpy as np
import pandas as pd
st.title('Active Volcanic Eruptions')
volc = 'C:/Users/hp/Desktop/Hassan/AUB/MSBA/Semester3/MSBA325/Assignments/Assignment2/volcano2.csv'
@st.cache
def load_data(nrows):
    data = pd.read_csv(volc, nrows=nrows)
    return data
data_load_state = st.text('Loading data...')
data = load_data(100)
data_load_state.text("Done! (using st.cache)")
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
st.text('Plotted on the map are the locations of active volcanos that have erupted in the past 10 years.\
    You can view the number and geographic location of these volcanos by year')
year_to_filter = st.slider('year', 2010, 2013,2018)
filtered_data = data[data.Year == year_to_filter]
st.subheader(f'Map of all volcanic eruptions  in {year_to_filter}')
st.map(filtered_data)

