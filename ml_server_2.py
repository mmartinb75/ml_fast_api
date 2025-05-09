from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from utils.transformations import ExtendedTransformation, SimpleTransformation
from utils.filters import SimpleFilter
from sklearn.ensemble import RandomForestRegressor
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    X: dict


app = FastAPI()
preproccesor = None
filter = None
model = None


@app.post("/predict/")
async def predict(request: PredictRequest):
    if any([m is None for m in [model, preproccesor, filter]]):
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        # transform X from json to dict
        x_dict = request.X
        x_pd = pd.DataFrame(x_dict)
        x_transform = preproccesor.transform(x_pd)
        x_filtered, _ = filter.transform(x_transform, None)
        y_pred, _ = model.predict(x_filtered)
        y_pred_un = preproccesor.inverse_transform(y_pred.reshape(-1, 1)).tolist()

        return {
            "y_pred": y_pred_un,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_w_intervals/")
async def predict(request: PredictRequest):
    if any([m is None for m in [model, preproccesor, filter]]):
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        # transform X from json to dict
        x_dict = request.X
        x_pd = pd.DataFrame(x_dict)
        x_transform = preproccesor.transform(x_pd)
        x_filtered, _ = filter.transform(x_transform, None)
        y_pred, intervals = model.predict(x_filtered)
        y_pred_un = preproccesor.inverse_transform(y_pred.reshape(-1, 1)).tolist()
        y_low = preproccesor.inverse_transform(intervals[:, 0]).tolist()
        y_up = preproccesor.inverse_transform(intervals[:, 1]).tolist()

        return {"y_pred": y_pred_un, "y_low": y_low, "y_up": y_up}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_model/")
async def upload_model(
    file_pre: UploadFile = File(...),
    file_filter: UploadFile = File(...),
    file_model: UploadFile = File(...),
):
    global model
    global preproccesor
    global filter
    if any(
        [not i.filename.endswith(".pkl") for i in [file_pre, file_filter, file_model]]
    ):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido.")

    content_pre = await file_pre.read()
    content_filter = await file_filter.read()
    content_model = await file_model.read()
    try:
        model = pickle.loads(content_model)
        preproccesor = pickle.loads(content_pre)
        filter = pickle.loads(content_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}")

    return {"status": "Modelo cargado correctamente"}
