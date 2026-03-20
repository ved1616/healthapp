from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("nutrisync_model.pkl")

class UserInput(BaseModel):
    Gender: str
    Age: int
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int
    Sleep_Disorder: str


def calculate_calories(d):
    if d["Gender"]=="Male":
        bmr=10*70+6.25*170-5*d["Age"]+5
    else:
        bmr=10*60+6.25*160-5*d["Age"]-161
    return int(bmr*(1+d["Physical Activity Level"]/10))


def calculate_macros(cal):
    return {
        "protein": int((cal*0.30)/4),
        "carbs": int((cal*0.40)/4),
        "fat": int((cal*0.30)/9)
    }


@app.post("/predict")
def predict(user: UserInput):

    data = {
        "Gender": user.Gender,
        "Age": user.Age,
        "Sleep Duration": user.Sleep_Duration,
        "Quality of Sleep": user.Quality_of_Sleep,
        "Physical Activity Level": user.Physical_Activity_Level,
        "Stress Level": user.Stress_Level,
        "BMI Category": user.BMI_Category,
        "Blood Pressure": user.Blood_Pressure,
        "Heart Rate": user.Heart_Rate,
        "Daily Steps": user.Daily_Steps,
        "Sleep Disorder": user.Sleep_Disorder
    }

    df = pd.DataFrame([data])

    cal = calculate_calories(data)
    macros = calculate_macros(cal)

    df["Calories Target"] = cal
    df["Protein (g)"] = macros["protein"]
    df["Carbohydrates (g)"] = macros["carbs"]
    df["Fat (g)"] = macros["fat"]

    pred = model.predict(df)[0]

    return {
        "health_category": pred,
        "calories": cal,
        "macros": macros
    }