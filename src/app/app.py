import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict


app = FastAPI(title="Lab Delay Prediction API")


@app.get("/")
def root():
    return {"status": "ok", "service": "lab-delay-prediction"}


@app.get("/health")
def health():
    return {"status": "ok"}


class LabData(BaseModel):
    labname: str
    gender: str
    age: str
    unittype: str
    recent_diagnosis: str | None = None
    result_year: int = 2026
    result_month: int = 1
    result_day: int = 1
    result_hour: int = 0
    result_weekday: int = 0
    admissionweight: float | None = None
    lab_workload_last_hour: int


@app.post("/predict")
def api_predict(data: LabData):
    return predict(data.model_dump())


LABNAME_CHOICES = [
    "potassium", "alkaline phos.", "Hgb", "calcium", "bicarbonate", "Hct",
    "creatinine", "-eos", "-monos", "-basos", "MCH", "MCHC", "-lymphs",
    "magnesium", "total bilirubin", "TSH", "anion gap", "albumin",
    "WBC x 1000", "ALT (SGPT)", "RDW", "AST (SGOT)", "RBC", "sodium",
    "MCV", "chloride", "total protein", "-polys", "glucose",
    "platelets x 1000", "BUN", "troponin - I", "lactate", "PT", "PT - INR",
    "PTT", "BNP", "bedside glucose", "Fe/TIBC Ratio", "TIBC", "Fe",
    "Ferritin", "reticulocyte count", "folate", "Vitamin B12",
    "Vancomycin - trough", "Digoxin", "free T4", "HCO3", "paCO2", "pH",
    "FiO2", "Base Deficit", "paO2", "LPM O2", "Base Excess",
    "total cholesterol", "LDL", "HDL", "triglycerides", "CRP",
    "urinary osmolality", "urinary specific gravity", "urinary sodium",
    "urinary creatinine", "uric acid", "serum osmolality", "ethanol",
    "salicylate", "Acetaminophen", "phosphate", "ionized calcium",
    "direct bilirubin", "amylase", "lipase", "CPK", "CPK-MB",
    "fibrinogen", "-bands"
]
GENDER_CHOICES = ["Male", "Female"]
UNITTYPE_CHOICES = ["Med-Surg ICU"]
DIAGNOSIS_CHOICES = [
    "None",
    "pulmonary|respiratory failure|acute respiratory distress",
    "endocrine|glucose metabolism|diabetes mellitus",
    "gastrointestinal|post-GI surgery|s/p surgery for intestinal obstruction",
    "cardiovascular|ventricular disorders|hypertension",
    "cardiovascular|chest pain / ASHD|coronary artery disease",
    "renal|disorder of kidney|chronic kidney disease|Stage 3 (GFR 30-59)",
    "pulmonary|disorders of the airways|COPD",
    "pulmonary|disorders of vasculature|pulmonary embolism",
    "toxicology|drug overdose|tricyclic overdose",
    "pulmonary|respiratory failure|acute respiratory failure",
]


def gradio_interface(
    labname,
    gender,
    age,
    unittype,
    recent_diagnosis,
    result_year,
    result_month,
    result_day,
    result_hour,
    result_weekday,
    admissionweight,
    lab_workload_last_hour,
):
    payload = {
        "labname": labname,
        "gender": gender,
        "age": str(age),
        "unittype": unittype,
        "recent_diagnosis": recent_diagnosis if recent_diagnosis != "None" else None,
        "result_year": int(result_year),
        "result_month": int(result_month),
        "result_day": int(result_day),
        "result_hour": int(result_hour),
        "result_weekday": int(result_weekday),
        "admissionweight": float(admissionweight) if admissionweight is not None else None,
        "lab_workload_last_hour": int(lab_workload_last_hour),
    }

    out = predict(payload)
    if out.get("status") != "success":
        return f"Error: {out.get('message', 'Unknown error')}"

    return (
        f"Delai predit : {out['predicted_turnaround_time_mins']} minutes\n"
        f"Periode predite : {out['predicted_period']}\n"
        f"Validation estimee : {out['predicted_validation_datetime']}"
    )


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(LABNAME_CHOICES, label="Nom du test labo", value="potassium"),
        gr.Dropdown(GENDER_CHOICES, label="Sexe", value="Female"),
        gr.Number(label="Age", value=78),
        gr.Dropdown(UNITTYPE_CHOICES, label="Unite hospitaliere", value="Med-Surg ICU"),
        gr.Dropdown(DIAGNOSIS_CHOICES, label="Diagnostic recent", value="None"),
        gr.Number(label="Annee de la demande", value=2026),
        gr.Number(label="Mois de la demande", value=1),
        gr.Number(label="Jour de la demande", value=1),
        gr.Number(label="Heure de la demande", value=8),
        gr.Number(label="Jour de semaine de la demande (0=lundi, 6=dimanche)", value=0),
        gr.Number(label="Poids du patient (kg)", value=65.5),
        gr.Number(label="Charge du labo durant la derniere heure", value=25),
    ],
    outputs="text",
    title="Predicteur des delais de laboratoire",
)

app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import socket
    import threading
    import webbrowser
    import uvicorn

    host = os.environ.get("APP_HOST", "0.0.0.0")
    start_port = int(os.environ.get("PORT", "8000"))

    def _find_available_port(port: int) -> int:
        for candidate in range(port, port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                    return candidate
        return port

    port = _find_available_port(start_port)
    print(f"Starting server on http://127.0.0.1:{port}/ui")

    if os.environ.get("OPEN_BROWSER", "1") == "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/ui")).start()

    uvicorn.run(app, host=host, port=port)
