# Test FastAPI web service for temperature prediction

import requests

# Define the API base URL
base_url = " https://temp-forecast-container-fegho6jisa-wn.a.run.app"  # Replace with the actual URL of your FastAPI service on GCP CLOUD RUN

# Test data
data = {
    "start_date": "2017-01-01",
    "end_date": "2018-01-01",
}

# Make a POST request to the forecast_temperature endpoint
response = requests.post(f"{base_url}/forecast_temperature/", json=data)

# Check the response status code
if response.status_code == 200:
    print("API call was successful")
    response_data = response.json()
    # You can now work with the response data as needed
    print(response_data)
else:
    print(f"API call failed with status code: {response.status_code}")
    print(response.text)
    