import pandas as pd
import numpy as np
from datetime import timedelta
from flask import Flask, jsonify, request
import statsmodels.api as sm

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    location_of_analysis = data['location']
    datetime_for_prediction = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M').floor('H')
    
    df = pd.read_csv('travio5.csv')
    
    df = df[df['Place'].str.contains(location_of_analysis)]
    df_exits = pd.read_csv('travio_exits.csv')
    df['datetime'] = pd.to_datetime(df['In Time']+' '+df['Date'], format='%H:%M %d-%m-%y')
    df['Hour'] = df['datetime'].dt.hour
    df['Date'] = df['datetime'].dt.date
    df['In Time'] = df['datetime'].dt.time
    df_exits['datetime'] = pd.to_datetime(df_exits['time']+' '+df_exits['date'], format='%H:%M %d-%m-%Y')
    df_sorted = df.sort_values('datetime')
    df_exits = df_exits.sort_values('datetime')

    df_sorted['datetime'] = pd.to_datetime(df_sorted['datetime']).dt.floor('H')
    df_exits['datetime'] = pd.to_datetime(df_exits['datetime']).dt.floor('H')
    df_sorted.drop_duplicates(inplace=True)
    df_exits.drop_duplicates(inplace=True)
    df_ts = df_sorted[['datetime', 'Total']].set_index('datetime')
    df_ts_exits = df_exits[['datetime', 'exits']].set_index('datetime')
    df_ts = df_ts.resample('H').sum().ffill().reset_index()
    df_ts_exits = df_ts_exits.resample('H').sum().ffill().reset_index()
    df_ts.set_index('datetime',inplace=True)
    df_ts_exits.set_index('datetime',inplace=True)
    intersecting_idx = df_ts_exits.index.intersection(df_ts.index)
    df_ts = df_ts.loc[intersecting_idx]
    df_ts_exits = df_ts_exits.loc[intersecting_idx]
    df_ts = df_ts['Total'].subtract(df_ts_exits['exits'])
    df_ts[df_ts < 0] = np.mean(df_ts)
    model = sm.tsa.statespace.SARIMAX(endog=df_ts,order=(1,0,1),seasonal_order=(1,0,1,24),freq='H')
    history = model.fit()
    print(history.summary())
    fc = history.predict(start = datetime_for_prediction, end=datetime_for_prediction+timedelta(hours=12))
    fc.index = fc.index.map(str)
    return jsonify(fc.to_dict())

@app.route('/',methods=['GET'])
def home():
    return '<p>API link is <a href="/predict">/predict</a></p> and POST data format is {"date" : "2022-03-02 11:54:00","location" : "Lingraj"}\n'

if __name__ == '__main__':
    app.run()
