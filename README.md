# 🌍 Health Forecasting Web Application
A powerful, AI-driven web application designed to forecast patient numbers and disease trends from historical health data — helping healthcare professionals and decision-makers plan smarter for the future.

## 🚀 Project Overview
This project harnesses the power of machine learning and time-series forecasting to provide accurate predictions of disease incidence and patient volume. With a user-friendly web interface and interactive visualizations, it empowers healthcare administrators, data analysts, and policymakers to make informed, data-driven decisions.

Built using FastAPI, Prophet, SARIMAX, and Plotly, the system simplifies complex modeling processes into a clean and responsive user experience.

## 💡 Key Features
### 🔮 Ensemble Forecasting
Combines Facebook Prophet and SARIMAX models for improved forecasting accuracy by averaging predictions from both methods.

### 📊 Interactive Data Visualizations
Generates dynamic bar and line charts using Plotly for clear, visual interpretation of future trends.

### 🧠 Data-Driven Decision Support
Forecasts disease or patient trends to support planning in areas such as hospital capacity, public health strategy, and supply allocation.

### 🌐 User-Friendly Interface
Built with FastAPI and Jinja2 templates to provide a simple form-based frontend for selecting features and forecast periods.

### 🔄 Cross-Origin Support
Integrated CORS middleware ensures compatibility with various frontend frameworks or client apps.

## 🧱 Application Architecture
The project is organized into the following components:

- main.py – Core FastAPI application handling routing, forecasting logic, and JSON responses

- template/index.html – Frontend HTML interface for user input and graph rendering

- Morbidity.csv – Cleaned and time-formatted dataset with historical patient/disease data

- static/ – Contains JavaScript files like plotly.min.js for frontend visualizations

- images/ – Optional folder for logos or branding (e.g., county or ministry logos)

## ⚙️ How It Works
- **User Input**
The user selects a disease category and specifies how many months to forecast.

- **Modeling**
Historical data is passed through both Prophet and SARIMAX models.

- **Ensemble Output**
Forecasts from both models are averaged for higher confidence in predictions.

- **Visualization**
Interactive graphs are rendered and served via the frontend interface.

## 🌍 Real-World Applications
This system is ideal for:

### 🏥 Hospitals & Clinics
Planning resources based on predicted patient volumes

### 🏛️ Public Health Agencies
Anticipating disease outbreaks and allocating response teams

### 📊 Health Data Analysts
Visualizing and interpreting health trends for reporting

### 🌐 Government & NGOs
Supporting evidence-based policy formulation in healthcare

## 📧 Contact
👤 Developed by: Gilbert Terer
📬 Email: teretich92@gmail.com
