import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import os
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler



matplotlib.use('Agg')
import seaborn as sns

st.set_page_config(layout="wide")
st.image("college.png")
st.header('Best College Institutions')


with st.sidebar:
  st.info("Select the dataset to learn more about it.")
  if st.checkbox("University Records"):
   st.write("The app allows users to explore and analyze a dataset of institution records and rankings. The dataset contains information about the institution's world rank, name, location, national rank, quality of education, alumni employment, quality of faculty, research performance, and score. Users can interact with the app to gain insights into the data by selecting different variables and filters. The app provides various features such as selecting specific institutions based on their national rank, filtering by location, and sorting by different rankings.")

#Read data
df = pd.read_csv('Uni_records.csv', encoding='latin-1')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Explanatory Analysis', 'Plots', 'Analysis','Predictive Modeling', 'Grid Search'])

with tab1:
    st.subheader("Look at the dataset")
    if st.checkbox("Show dataset"):
        number = st.number_input("Number of rows to view",5,100)
        st.dataframe(df.head(number))

    st.subheader("Filter Columns to Compare and Contrast")
    if st.checkbox("Select Columns to Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    st.subheader("Look at the Summary")
    if st.checkbox("Summary"):
        st.write(df.describe())

 
with tab2: 
    st.subheader ("Data Viz")

    if st.checkbox("Correlation Plot(Seaborn)"):
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot",["bar", "area", "line","hist"])
    selected_columns_names = st.multiselect("Select Columns to plot", all_columns_names)

    if st.checkbox("Plot of Value counts"):
        st.text("Value counts by target")
    all_columns_names = df.columns.tolist()
    primary_col = st.selectbox("Primary column to Groupby", all_columns_names)
    selected_columns_names = st.multiselect("Select Columns", all_columns_names)
    if st.button("Plot"):
      st.text("Generated Plot")
      if selected_columns_names:
         vc_plot = df.groupby(primary_col)[selected_columns_names].count()
      else:
         vs_plot = df.iloc[:,1].value_counts()
         st.write(vs_plot.plot(kind="bar"))
         st.pyplot()
    if st.button("Generate Plot"):
        st.success("Generate custom plot of {} for {}".format(type_of_plot,selected_columns_names))

    if type_of_plot == 'area':
        cust_data = df[selected_columns_names]
        st.area_chart(cust_data)
    elif type_of_plot == 'bar':
        cust_data = df[selected_columns_names]
        st.bar_chart(cust_data)
    elif type_of_plot == 'line':
        cust_data = df[selected_columns_names]
        st.line_chart(cust_data)
    elif type_of_plot: 
        cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

   
