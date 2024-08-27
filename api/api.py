import pandas as pd
import numpy as np
import sklearn
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

sklearn.set_config(transform_output='pandas')

app = FastAPI()

model_loaded = joblib.load('./best_model.pkl')

class LogData(BaseEstimator, TransformerMixin):

    def __init__(self, col_to_log):
        self.col_to_log = col_to_log

    def transform(self, X):
        for col in X.columns:
            if col in self.col_to_log:
                X.loc[:, col] = np.log(1 + X.loc[:, col])
        return X

    def fit(self, X, y=None):
        return self


class Dataframe(BaseModel):
    data: str


@app.post("/best_model")
async def best_model(one_var: Dataframe):

    df = pd.read_json(one_var.data, orient='split')
    pred = model_loaded.predict(df)
    pred = pd.DataFrame(pred)
    return {"pred": pred.to_json(orient='split')}


