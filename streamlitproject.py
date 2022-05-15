from nbformat import write
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Price Predictor", "Visualizations"],
                         icons=['house', 'piggy-bank-fill', 'graph-up'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
df = pd.read_csv('merc.csv')
import streamlit as st
#st.set_page_config('Mercedes Price Prediction App')
#st.title('Predict Mercedes Car Prices (in Euros)')
#menu_list = ['Visualization', "Predict Price"]
#menu = st.radio("Menu", menu_list)
if choose =="About":
    st.title("Mercedes Hub")
    st.write("This app will help you better price the cars you sell by allowing predictions of price based on historic sales data. The price predictor allows custom picking of car specs like model manufacture year, model type, engine size, etc.. It then predicts the price by running your choice of a Decision Tree or Linear Regresson Models.")
    st.write("The data is cleaned and preprocessed automatically through the pipeline")




elif choose == 'Visualizations':
    st.title('Exploratory Data Analysis of Mercedes Benz Car Models ')
    col1,col2=st.columns(2)

    with col1:
        if st.checkbox("View data"):
            st.write(df)
    with col2:
        if st.checkbox("Price Vs Model"):
            plt.figure(figsize=(20,8),dpi=80)
            ax=plt.axes()
            ax.set_facecolor('Black')
            plt.bar(df['model'],df['price'],color='white')
            plt.scatter(df['model'],df['price'],s=100,color='red')
            plt.title('Model Vs. Price',color='black',fontsize=25)
            plt.xticks(rotation=50,color='black')
            plt.yticks(color='black')
            plt.xlabel('Model',color='black',fontsize=20)
            plt.ylabel('Model Price',color='black',fontsize=20)
            graph1 = plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(graph1)
    with col1:
        if st.checkbox("Dominant Fuel Type"):
            fuels = pd.DataFrame(df['fuelType'].value_counts()) 
            group = df.groupby(df['fuelType']) 
            mean_price = pd.DataFrame(group.price.mean())
            fuels.reset_index(level=0, inplace=True) 
            fuels.columns = ('fuelType', 'size') 
            fuels = pd.merge(fuels,mean_price,how='left', on='fuelType') 
            labels = ["%s\n%d items\nMean price: %d€"% (label) for label in 
                zip(fuels['fuelType'], fuels['size'], fuels['price'])] 
            fig1, ax1 = plt.subplots() 
            ax1.pie(fuels['size'], labels=labels, 
                autopct='%1.1f%%', startangle=50, colors=plt.cm.Set1.colors) 
            ax1.axis('equal') 
            plt.title("Percentage of Fuels ")
            graph2 = plt.show()
            st.pyplot(graph2)
    with col2:
        if st.checkbox("Engine Size per Model"):
            m = df['model']
            s = df['engineSize']
            plt.stem(m, s)
            plt.xticks(rotation=90)
            plt.title('Engine Size')
            graph3 = plt.show()
            st.pyplot(graph3)
    with col1:
        if st.checkbox("MPG by Model"):
            graph4 = sns.barplot(x="mpg", y="model", data=df)
            plt.title("Miles per gallon")
            st.pyplot(graph4.figure)

elif choose == 'Price Predictor':
    st.title("Predict The Price of Your Mercedes")

    model_dic = {'a class': 0, 'b class': 1, 'c class': 2, 'cl class': 3, 'cla class': 4, 'clc class': 5, 'clk': 6,
                 'cls class': 7, 'e class': 8, 'g class': 9, 'gl class': 10, 'gla class': 11, 'glb class': 12,
                 'glc class': 13, 'gle class': 14, 'gls class': 15, 'm class': 16, 'r class': 17, 's class': 18,
                 'sl class': 19, 'slk': 20, 'v class': 21, 'x-class': 22}
    transmission_dic = {'automatic': 0, 'manual': 1, 'other': 2, 'semi-auto': 3}
    fuel_dic = {'diesel': 0, 'hybrid': 1, 'other': 2, 'petrol': 3}

    model_list = [
        "a class", "b class", "c class", "cl class", "cla class", "clc class", "clk", "cls class", "e class", "g class",
        "gl class", "gla class", "glb class", "glc class", "gle class", "gls class", "m class", "r class", "s class",
        "sl class", "slk", "v class", "x-class"]
    transmission_list = ['automatic', 'manual', 'other', 'semi-auto']
    fuel_list = ['diesel', 'hybrid', 'other', 'petrol']

    year = st.slider("Enter the year", 1970, 2021)

    engine_size = st.number_input('Enter Engine Size  (range = 0 - 7)')

    model_choice = st.selectbox(label='Select Car Model', options=model_list)
    models = model_dic[model_choice]

    transmission_choice = st.selectbox(label=' Select Transmission Type', options=transmission_list)
    transmissions = transmission_dic[transmission_choice]

    fuel_choice = st.selectbox(label='Select Fuel Type', options=fuel_list)
    fuels = fuel_dic[fuel_choice]
    data = pd.read_csv('merc.csv')
    data['models'] = data['model'].str.strip()
    df = data.drop('model', axis='columns')

    OH_encoder = OneHotEncoder(sparse=False)
    encode_data = pd.DataFrame(OH_encoder.fit_transform(df[['transmission', 'fuelType']]))

    encode_data.columns = ['Automatic', 'Manual', 'Other', 'Semi-Auto', 'Diesel', 'Hybrid', 'Other', 'Petrol']

    merc_data = pd.concat([df, encode_data], axis=1)
    df1 = merc_data.drop(['transmission', 'fuelType', 'models'], axis='columns')
    df2 = pd.get_dummies(df.models)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['price', 'tax', 'mpg', 'mileage'], axis='columns')
    y = df3.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    decision_tree = DecisionTreeRegressor()
    linear_reg = LinearRegression()

    decision_tree.fit(X_train.values, y_train.values)
    linear_reg.fit(X_train.values, y_train.values)

    decision_score = decision_tree.score(X_test.values, y_test.values)
    linear_score = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values


    def predict_price_decision(model, _year, engineSize, transmission, fuel):
        try:
            model_index = model_list.index(model)[0][0]
            transmission_index = transmission_list.index(transmission)[0][0]
            fuel_index = fuel_list.index(fuel)[0][0]
        except ValueError:
            model_index = -1
            fuel_index = -1
            transmission_index = -1

        x = np.zeros(len(column_data))
        x[0] = _year
        x[1] = engineSize
        if transmission_index >= 0:
            x[transmission_index] = 1
        elif fuel_index >= 0:
            x[fuel_index] = 5
        elif model_index >= 0:
            x[model_index] = 9

        return decision_tree.predict([x])[0]

    def predict_price_linear(model, _year, engineSize, transmission, fuel):
        try:
            model_index = model_list.index(model)[0][0]
            transmission_index = transmission_list.index(transmission)[0][0]
            fuel_index = fuel_list.index(fuel)[0][0]
        except ValueError:
            model_index = -1
            fuel_index = -1
            transmission_index = -1

        x = np.zeros(len(column_data))
        x[0] = _year
        x[1] = engineSize
        if transmission_index >= 0:
            x[transmission_index] = 1
        elif fuel_index >= 0:
            x[fuel_index] = 5
        elif model_index >= 0:
            x[model_index] = 9

        return linear_reg.predict([x])[0]


    alg = ['Decision Tree Regression', 'Linear Regression']
    select_alg = st.selectbox('Choose Predictor', alg)
    if st.button('Predict Price'):
        if select_alg == 'Decision Tree Regression':
            st.write('Accuracy Score', decision_score)
            st.subheader(predict_price_decision(models, year, engine_size, transmissions, fuels))
            st.markdown("<h5 style='text-align: left;'> Euros </h5>", unsafe_allow_html=True)

        elif select_alg == 'Linear Regression':
            st.write('Accuracy Score', linear_score)
            predicted_price = st.subheader(predict_price_linear(models, year, engine_size, transmissions, fuels))

        

