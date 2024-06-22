import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from flask import Flask, request, render_template
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            airline=request.form.get('airline'),
            source_city=request.form.get('source_city'),
            destination_city=request.form.get('destination_city'),
            departure_time=request.form.get('departure_time'),
            arrival_time=request.form.get('arrival_time'),
            Class=request.form.get('Class'),
            stops=request.form.get('stops'),
            days_left=int(request.form.get('days_left')),
            duration=float(request.form.get('duration'))
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)