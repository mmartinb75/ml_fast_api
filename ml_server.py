from fastapi import FastAPI, UploadFile
from utils.transformations import ExtendedTransformation, SimpleTransformation
from utils.filters import SimpleFilter
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()


@app.get("/hello")
async def root():
    return {"message": "Bienvenido a mi Aplicación de Machine Learning server"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
