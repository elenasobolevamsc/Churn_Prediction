import pandas as pd
import numpy as np
import sklearn
import joblib
import clickhouse_connect
import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from lib.custom_classes_for_pipeline import LogData


sklearn.set_config(transform_output='pandas')

app = FastAPI()

model_loaded = joblib.load('./best_model.pkl')



class Dataframe(BaseModel):
    data: str


@app.post("/best_model")
async def best_model(one_var: Dataframe):
    client = clickhouse_connect.get_client(host='158.160.29.135', port=8123, username='default', password='7333734')
    name = f'tmp.client_{int(datetime.datetime.now().timestamp()*10000)}'
    client.command(f"""DROP TABLE IF EXISTS {name}""")
    client.command(f"""CREATE TABLE {name} 
    (
    `customerID` String
    )
    ENGINE = MergeTree()
    ORDER BY customerID""")


    df = pd.read_json(one_var.data, orient='split')
    id = df[['customerID']]
    client.insert(name, id)

    data = client.query_df(f"""SELECT * FROM telecom.contract c
                                JOIN telecom.internet i on i.customerID = c.customerID
                                JOIN telecom.phone p on p.customerID = c.customerID
                                JOIN telecom.personal per on per.customerID = c.customerID
                                WHERE c.customerID IN (SELECT t.customerID FROM {name} t)
                                """)
    data['Churn'] = data['EndDate'].apply(lambda x: 0 if x == 'No' else 1)
    data['EndDate'] = data['EndDate'].apply(lambda x: '2020-02-01' if x == 'No' else x)
    data['EndDate'] = pd.to_datetime(data['EndDate'], format='%Y-%m-%d')
    data['Lifetime'] = (
            data['EndDate'].apply(lambda x: x.tz_localize(None))
            - data['BeginDate'].apply(lambda x: x.tz_localize(None))
    ).dt.days

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors ='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(0)

    data = data.drop(['BeginDate',
                      'EndDate',
                      'gender',
                      'Dependents',
                      'PaperlessBilling',
                      'TechSupport',
                      'OnlineSecurity',
                      'i.customerID',
                      'p.customerID',
                      'per.customerID'], axis=1)
    data = data.rename(columns={'c.customerID': 'customerID'})
    data = data.set_index('customerID')
    data = data.fillna('No')

    pred = model_loaded.predict(data)
    pred = pd.DataFrame(pred)
    return {"pred": pred.to_json(orient='split')}


