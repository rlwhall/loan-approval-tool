from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle

# create API
app = Flask(__name__)
api = Api(app)

instructions = "\n Welcome! <br> please post your data to the '/predict' endpoint to recieve a prediction"

class Greet(Resource):
  def get(self):
    return jsonify(instructions)

# assign endpoint
api.add_resource(Greet, '/')

# bring in class used during model creation
# functions of class are stored in model
class ToDenseTransformer():
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
    def fit(self, X, y=None, **fit_params):
        return self

def cr_obj(data):
    data['Credit_History'] = data.Credit_History.astype('object')
    return data
    
def numFeat(data):
    num_feats = data.select_dtypes(['int','float']).columns.tolist()
    return data[num_feats]

def catFeat(data):
    cat_feats = data.select_dtypes('object').columns.tolist()
    return data[cat_feats]

def create_total(data):
    data['total_income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    return data

def log_transform(data):
    data_df = pd.DataFrame(data)
    cols = [3,5]
    for col in cols:
        data_df[col] = np.log(data_df[col])
    data_np = data_df.to_numpy()
    return data_np

dtypes={'Gender': object, 'Married': object, 'Dependents': object, 
        'Education': object,'Self_Employed':object,'ApplicantIncome': float, 'CoapplicantIncome': float,
       'LoanAmount': float, 'Loan_Amount_Term': float, 'Credit_History': float, 'Property_Area': object}

# load model
model = pickle.load(open('loan_prediction_model.sav', 'rb'))

# create endpoint for predict
class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        df = df.astype(dtype=dtypes)
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict_proba(df)
        loan_status = model.predict(df)
        # we cannot send numpy array as a result, so return list
        return loan_status.tolist() 

# assign endpoint
api.add_resource(Predict, '/predict')

# create application run when file is run directly
if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5555)
