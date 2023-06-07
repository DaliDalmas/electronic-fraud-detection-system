import pickle as pk
import pandas as pd
import joblib

filehandler_1 = open(b"lab/models/decision_trees_2.pkl","rb")
type_label_encoder = joblib.load(filehandler_1)
filehandler_1.close()
filehandler_2 = open(b"lab/models/type_label_encoder.pkl","rb")
decision_tree_model = joblib.load(filehandler_2)
filehandler_2.close()

class MlModel:
    def __init__(self):
        pass
    def predict(self, transaction):
        transaction['transformed_type'] = type_label_encoder.transform(
            transaction['type']
            )
        del transaction['type']
        transaction = pd.DataFrame([transaction])

        return {
            "verdict": 'fraud' if decision_tree_model.predict(transaction)==1 else 'not_fraud',
            
            "verdict_proba": decision_tree_model.predict_proba(transaction)
        }



        
        
