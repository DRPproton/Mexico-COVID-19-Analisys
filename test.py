import requests

patient = {'sexo': 1.0,
 'neumonia': 1.0,
 'edad': 75,
 'diabetes': 1.0,
 'epoc': 1.0,
 'inmusupr': 0.0,
 'hipertension': 1.0,
 'cardiovascular': 1.0,
 'obesidad': 1.0,
 'renal_cronica': 0.0}


host = "https://mexico-covid-prediction.onrender.com"

# url = "http://localhost:9696/predict"
# url = "http://127.0.0.1:8000/predict"

url = f"{host}/predict"

response = requests.post(url, json=patient).json()

print(response)
