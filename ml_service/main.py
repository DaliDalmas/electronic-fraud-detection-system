from fastapi import FastAPI
from pydantic import BaseModel
from ml_service.ml_model import MlModel
app = FastAPI()


class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    cash_in_mean: float
    cash_out_mean: float
    debit_mean: float
    payment_mean: float
    transfer_mean: float
    cash_in_min: float
    cash_out_min: float
    debit_min: float
    payment_min: float
    transfer_min: float
    cash_in_max: float
    cash_out_max: float
    debit_max: float
    payment_max: float
    transfer_max: float
    cash_in_count: float
    cash_out_count: float
    debit_count: float
    payment_count: float
    transfer_count: float
    type: str

@app.post('/fraud_check_this_transaction/')
def check_fraud(transaction: Transaction):
    print(transaction.dict())
    return MlModel().predict(transaction = transaction.dict())