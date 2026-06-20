import requests

url = "http://127.0.0.1:8000/predict"

# Sample lab order data matching your lab_pred.csv schema
sample_data = {
    "labid": 12345678,
    "labname": "potassium",
    "result_time": 360,           # minutes offset (e.g., 6 hours from reference)
    "validation_time": 400,       # minutes offset (will be ignored by model)
    "gender": "Female",
    "age": "78",
    "unittype": "Med-Surg ICU",
    "admissionweight": 65.5,
    "recent_diagnosis": "cardiovascular|ventricular disorders|hypertension",
    "lab_workload_last_hour": 25
}

response = requests.post(url, json=sample_data)
print("Status Code:", response.status_code)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.text)