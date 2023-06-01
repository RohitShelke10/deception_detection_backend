from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import librosa
import io
from tensorflow import keras
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MODEL = keras.models.load_model('./LSTM_sigmoid/')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Lie Detector"})

@app.post("/")
async def get_results(sound: bytes=File(...)):
    y, sr = librosa.load(io.BytesIO(sound))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17)
    mfcc = mfcc.T
    print(mfcc.shape)
    mfcc = np.array(mfcc)
    mfcc = mfcc.reshape((mfcc.shape[0],1,mfcc.shape[1]))
    prediction = MODEL.predict(mfcc)
    prediction_probability = np.mean(prediction.ravel())*100

    return JSONResponse(content={"prediction": prediction_probability})
