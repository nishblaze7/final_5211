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
df = pd.read_csv('Uni_records.csv')
st.write(df)


   
