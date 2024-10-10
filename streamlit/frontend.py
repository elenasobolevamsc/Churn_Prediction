import streamlit as st
import pandas as pd
import requests
from io import StringIO

UI_API = 'api-churn-prediction'

# Define a function that handles the API request
def send_request(file):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Convert DataFrame to JSON (this depends on your API requirements)
    data_json = df.to_json(orient='split')
    payload = {'data': data_json}

    # Example API endpoint (replace with your actual endpoint)
    url = f"http://{UI_API}/best_model"

    # Send POST request to the API (you may need to adjust headers and data format)
    response = requests.post(url, json=payload)

    res = pd.read_json(response.json()['pred'], orient='split')
    res.columns = ['Prediction']
    pred_res = pd.concat([df, res], axis=1)

    return pred_res

# Streamlit app
st.title('Churn Prediction App')

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Display uploaded file as a DataFrame
    st.write("Uploaded Data:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Predict button
    if st.button('Predict'):
        with st.spinner('Sending request to API...'):
            # Send the file to the API and get the predictions
            result = send_request(uploaded_file)

            # Display the processed DataFrame with predictions
            st.write("Prediction Results:")
            st.dataframe(result)

            # Provide download link for processed file
            output_file = "processed_file.csv"
            result.to_csv(output_file, index=False)

            # Create a download button for the CSV
            st.download_button(
                label="Download CSV with Predictions",
                data=result.to_csv(index=False),
                file_name="processed_file.csv",
                mime="text/csv"
            )
