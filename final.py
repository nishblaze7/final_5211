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

   
with tab3:    
    st.subheader('Top Institutions By Country')
    country = st.selectbox('Select a country', df['Location'].unique())
    filtered_df = df[df['Location'] == country]

    if len(filtered_df) == 0:
        st.write(f"No data available for {country}. Please select another country.")
    else:
        n = 10  # Number of institutions to display
        top_institutions = filtered_df.sort_values(by='World Rank', ascending=True).head(n)

        st.write(f"Top {n} institutions in {country} with high world rankings:")
        st.table(top_institutions[['Institution', 'World Rank']])

    st.subheader('Regression')

    independent_vars = ['Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', 'Research_Performance']
    selected_vars = st.multiselect('Select independent variables', independent_vars, default=independent_vars)
    dependent_var = 'Score'

    selected_cols = selected_vars + [dependent_var]
    data = df[selected_cols]

    X = sm.add_constant(data[selected_vars])
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()

    st.write(model.summary())
    

with tab4:
    st.title('Decision Tree Regression')
    variable_names = ['Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', 'Research_Performance']
    selected_vars = st.multiselect('Select variables:', variable_names, default=variable_names)
    X = df[selected_vars]
    y = df['Score']
    st.write('Select variables and evaluate a decision tree model.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write('Mean squared error:', mse)

    st.write('Decision Tree:')
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, ax=ax, feature_names=X.columns, fontsize=10)
    st.pyplot(fig)    
    
with tab5:
    X = df[['Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', 'Research_Performance']]
    y = df['Score']

    k_range = range(1, 21)
    k = st.slider('Select a value of k:', min_value=1, max_value=20)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)

    mse = mean_squared_error(y, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'K-Nearest Neighbors Regression (k={k})\nMSE: {mse:.2f}')
    st.pyplot(fig)
    
