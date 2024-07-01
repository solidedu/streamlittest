import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data from a JSON file
@st.cache_data
def load_data():
    data = pd.read_json('data.json')
    data['date'] = pd.to_datetime(data['date'])
    return data

# Function to calculate summary metrics
def calculate_metrics(data):
    total_value1 = data['value1'].sum()
    total_value2 = data['value2'].sum()
    avg_value1 = data['value1'].mean()
    avg_value2 = data['value2'].mean()
    return total_value1, total_value2, avg_value1, avg_value2

# Main function to run the Streamlit app
def main():
    st.title('Enhanced Data Visualization with Streamlit')

    # Load the data
    data = load_data()

    st.write("Here's a glimpse of the data:")
    st.write(data.head())

    # Sidebar for user inputs
    st.sidebar.header('User Input')
    text_input = st.sidebar.text_input('Enter a value:')
    st.sidebar.write(f'You entered: {text_input}')
    
    if text_input:
        st.write(f'Text Input: {text_input}')

    st.sidebar.header('Visualization Options')
    viz_type = st.sidebar.selectbox('Select Visualization Type:', ['Scatter Plot', 'Histogram', 'Line Chart'])

    # Visualization based on user selection
    if viz_type == 'Scatter Plot':
        st.write("## Scatter Plot")
        x_axis = st.sidebar.selectbox('Select X-axis:', data.columns)
        y_axis = st.sidebar.selectbox('Select Y-axis:', data.columns)
        fig, ax = plt.subplots()
        ax.scatter(data[x_axis], data[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f'Scatter plot of {x_axis} vs {y_axis}')
        st.pyplot(fig)

    elif viz_type == 'Histogram':
        st.write("## Histogram")
        column = st.sidebar.selectbox('Select column for histogram:', data.columns)
        bins = st.sidebar.slider('Number of bins:', min_value=10, max_value=100, value=30)
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=bins)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
        st.pyplot(fig)

    elif viz_type == 'Line Chart':
        st.write("## Line Chart")
        x_axis = st.sidebar.selectbox('Select X-axis:', data.columns)
        y_axis = st.sidebar.selectbox('Select Y-axis:', data.columns)
        fig, ax = plt.subplots()
        ax.plot(data[x_axis], data[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f'Line chart of {x_axis} vs {y_axis}')
        st.pyplot(fig)

    # Filter data functionality
    st.write("## Filter Data")
    column_to_filter = st.selectbox('Select column to filter:', data.columns)
    filter_value = st.text_input(f'Enter value to filter {column_to_filter} by:')
    filtered_data = data[data[column_to_filter].astype(str).str.contains(filter_value, case=False, na=False)]

    st.write("Filtered Data:")
    st.write(filtered_data)

    # Calculate and display summary metrics
    st.write("## Summary Metrics")
    total_value1, total_value2, avg_value1, avg_value2 = calculate_metrics(data)
    st.metric("Total Value 1", total_value1)
    st.metric("Total Value 2", total_value2)
    st.metric("Average Value 1", avg_value1)
    st.metric("Average Value 2", avg_value2)

    # Dashboard layout
    st.write("## Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Line Chart of Value 1")
        fig, ax = plt.subplots()
        ax.plot(data['date'], data['value1'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Value 1')
        ax.set_title('Value 1 over Time')
        st.pyplot(fig)

    with col2:
        st.write("### Line Chart of Value 2")
        fig, ax = plt.subplots()
        ax.plot(data['date'], data['value2'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Value 2')
        ax.set_title('Value 2 over Time')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
