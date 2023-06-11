import pandas as pd
import pickle
import numpy as np


class MlModel:

    def __init__(self):
        with open(b"./lab/models/decision_trees_2.pkl","rb") as file:  
            self.decision_tree_model = pickle.load(file)
        with open(b"./lab/models/type_label_encoder.pkl","rb") as file:  
            self.type_label_encoder = pickle.load(file)

    def predict(self, transaction):
        transaction['transformed_type'] = self.type_label_encoder.transform(
            np.array([transaction['type']])
            )
        del transaction['type']
        transaction = pd.DataFrame([transaction])

        return {
            "verdict": 'fraud' if self.decision_tree_model.predict(transaction)==1 else 'not_fraud',
            
            "verdict_proba": self.decision_tree_model.predict_proba(transaction)
        }
