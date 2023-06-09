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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression




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


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Explanatory Analysis', 'Plots', 'Analysis','Predictive Modeling', 'Grid Search','Comparing models'])

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
    if st.checkbox("Show Feature Importance"):
        # select the independent variables and dependent variable
        X = df[['Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', 'Research_Performance']]
        y = df['Score']

        model = RandomForestRegressor()
        model.fit(X, y)

        importances = model.feature_importances_

        fig, ax = plt.subplots()
        ax.bar(X.columns, importances)
        plt.xticks(rotation=45)
        ax.set_title('Feature Importances')
        st.pyplot(fig)

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
    if st.checkbox("Select to learn more about the following Regression"):
        st.write("Please adjust your independent variables, which will be then used to create the X variable, which represents the independent variables in the linear regression analysis. The y variable [Score] represents the dependent variable. The model summary provides a summary of the regression analysis results. This includes information on the coefficients of the independent variables, which indicate how strongly each independent variable is associated with the dependent variable. If the R-squared value is close to 1, it means that the independent variables are very good at predicting the dependent variable. If the R-squared value is close to 0, it means that the independent variables are not very good at predicting the dependent variable.")


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
    if st.checkbox("Select to learn more about the following Decision Tree Analysis"):
        st.write("This code is performing a decision tree regression analysis on a dataset with four independent variables (Quality_of_Education, Alumni_Employment, Quality_of_Faculty, and Research_Performance) and one dependent variable (Score). The model will split the data into training and testing sets. The decision tree regression model is then fit to the training data. This will generate predicted values for the test data, and the mean squared error, which is used to calculate the mean squared error of the predicted values compared to the actual values. The graphical representation of the decision tree can also help to visualize the decision-making process of the model.")


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
    st.subheader("K-Means Regression")
    if st.checkbox("Select to learn more about the following K-means regression"):
        st.write("K-Nearest Neighbors Regression using the 'Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', and 'Research_Performance' variables to predict the 'Score' of a university.The slider allows you to choose a value of k (number of nearest neighbors to use in the regression) and see the resulting scatter plot of measured vs. predicted values. The black dashed line represents a perfect match between the measured and predicted values, and the blue dots represent the actual data points. The mean squared error (MSE) is also displayed on the plot.")
                 

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

with tab6:
    X = df[['Quality_of_Education', 'Alumni_Employment', 'Quality_of_Faculty', 'Research_Performance']]
    y = df['Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create and fit a linear regression model
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    linear_reg_y_pred = linear_reg.predict(X_test)

# create and fit a decision tree regression model
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)
    dt_reg_y_pred = dt_reg.predict(X_test)

# create and fit a random forest regression model
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    rf_reg_y_pred = rf_reg.predict(X_test)

# calculate and print the evaluation metrics for each model
    linear_reg_mse = mean_squared_error(y_test, linear_reg_y_pred)
    linear_reg_r2 = r2_score(y_test, linear_reg_y_pred)

    dt_reg_mse = mean_squared_error(y_test, dt_reg_y_pred)
    dt_reg_r2 = r2_score(y_test, dt_reg_y_pred)

    rf_reg_mse = mean_squared_error(y_test, rf_reg_y_pred)
    rf_reg_r2 = r2_score(y_test, rf_reg_y_pred)

    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
        'MSE': [linear_reg_mse, dt_reg_mse, rf_reg_mse],
        'R-squared': [linear_reg_r2, dt_reg_r2, rf_reg_r2]})

# display the results in a table using streamlit
    st.write('Regression Models Evaluation Metrics:')
    st.write(results)
