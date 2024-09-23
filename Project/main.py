from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import plotly.express as px
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

templates = Jinja2Templates(directory="template")

# Load the data (example data)
df = pd.read_csv('Morbidity.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)
df = df.asfreq('MS')

# Define the ensemble forecasting function
def ensemble_forecast(df, feature, forecast_months=12):
    # Prepare data for Prophet
    df_feature = df[[feature]].reset_index()
    df_feature.columns = ['ds', 'y']
    
    # Prophet Model
    model_prophet = Prophet(seasonality_prior_scale=10.0)
    model_prophet.fit(df_feature)
    future_prophet = model_prophet.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast_prophet = model_prophet.predict(future_prophet)
    forecast_future_prophet = forecast_prophet[forecast_prophet['ds'] > df_feature['ds'].max()]
    
    # Prepare data for SARIMAX
    df_sarimax = df[feature]
    
    # SARIMAX Model
    model_sarimax = sm.tsa.statespace.SARIMAX(df_sarimax, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results_sarimax = model_sarimax.fit()
    forecast_sarimax = results_sarimax.get_forecast(steps=forecast_months)
    forecast_future_sarimax = forecast_sarimax.predicted_mean.reset_index()
    forecast_future_sarimax.columns = ['ds', 'yhat']
    forecast_future_sarimax['ds'] = pd.date_range(start=df_sarimax.index.max() + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    
    # Combine forecasts using simple average
    forecast_future = forecast_future_prophet[['ds', 'yhat']].copy()
    forecast_future['yhat_sarimax'] = forecast_future_sarimax['yhat'].values
    forecast_future['yhat_ensemble'] = (forecast_future['yhat'] + forecast_future['yhat_sarimax']) / 2
    
    # Round predictions to whole numbers
    forecast_future['yhat_ensemble'] = forecast_future['yhat_ensemble'].round().astype(int)
    
    # Convert Timestamp to string for JSON serialization
    forecast_future['ds'] = forecast_future['ds'].dt.strftime('%Y-%m-%d')
    
    return forecast_future[['ds', 'yhat_ensemble']]

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    features = df.columns.tolist()
    return templates.TemplateResponse("index.html", {"request": request, "features": features})

@app.post("/forecast", response_class=JSONResponse)
async def forecast(feature: str = Form(...), months: int = Form(...)):
    forecast_data = ensemble_forecast(df, feature, months)

    # Debug message
    debug_message = f"Received feature: {feature}, Forecast months: {months}"
    print(debug_message)

    # Plotly bar graph
    bar_fig = px.bar(forecast_data, x='ds', y='yhat_ensemble', title=f'Forecast for {feature} - Bar Graph')
    bar_fig.update_layout(xaxis_title='Months',
                         yaxis_title='Number of Patients')
    bar_graph = bar_fig.to_json()
        
    # Plotly line graph
    line_fig = px.line(forecast_data, x='ds', y='yhat_ensemble', title=f'Forecast for {feature}- Line Graph')
    line_fig.update_layout(xaxis_title='Months',
                          yaxis_title='Number of Patients')
    line_graph = line_fig.to_json()

    return JSONResponse(content={
        'bar_graph': bar_graph,
        'line_graph': line_graph,
        'forecast_data': forecast_data.to_dict(orient='records'),
        'debug_message': debug_message
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)