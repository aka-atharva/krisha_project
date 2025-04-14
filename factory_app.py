import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime, timedelta

# Configuration
st.set_page_config(page_title="Factory Predictive System", layout="wide")
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Expected data format
REQUIRED_COLUMNS = [
    'Timestamp', 'MachineID', 'Temperature', 'Vibration', 
    'Pressure', 'OperatingHours', 'MaintenanceFlag',
    'ProductQuality', 'UnitsProduced'
]

# Threshold configuration
PARAM_CONFIG = {
    'Temperature': {'unit': '°C', 'warning': (60, 75), 'critical': 80},
    'Vibration': {'unit': 'mm/s', 'warning': (1.5, 2.5), 'critical': 3.0},
    'Pressure': {'unit': 'psi', 'warning': (150, 180), 'critical': 200}
}

QUALITY_THRESHOLDS = {
    'Excellent': 0.9,
    'Good': 0.7,
    'Fair': 0.5,
    'Poor': 0.0
}

PRODUCTION_TARGET = 100  # Daily production target per machine

def load_real_data():
    """Load and preprocess real factory data"""
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not data_files:
        st.error(f"No CSV files found in {DATA_DIR} directory")
        st.info("Please upload your machine data CSV files")
        return None
    
    dfs = []
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            if not all(col in df.columns for col in REQUIRED_COLUMNS):
                st.error(f"Missing required columns in {file}")
                continue
                
            # Convert timestamp and sort
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            dfs.append(df.sort_values('Timestamp'))
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        return None
    
    return pd.concat(dfs)

def train_models(data):
    """Train models for maintenance, quality and production"""
    models = {}
    
    # Feature engineering
    data['TimeSinceMaintenance'] = data.groupby('MachineID')['MaintenanceFlag']\
        .cumsum().groupby(data['MachineID']).cumcount()
    data['QualityClass'] = (data['ProductQuality'] >= QUALITY_THRESHOLDS['Good']).astype(int)
    
    # Common features
    features = ['Temperature', 'Vibration', 'Pressure', 'OperatingHours', 'TimeSinceMaintenance']
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    
    # 1. Maintenance models
    for target in ['Temperature', 'Vibration', 'Pressure']:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, data[target])
        models[f'maintenance_{target}'] = model
    
    # Maintenance classifier (will machine need maintenance soon?)
    maint_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    maint_clf.fit(X, data['MaintenanceFlag'])
    models['maintenance_classifier'] = maint_clf
    
    # 2. Quality model
    quality_model = RandomForestClassifier(n_estimators=100, random_state=42)
    quality_model.fit(X, data['QualityClass'])
    models['quality'] = quality_model
    
    # 3. Production model
    production_model = RandomForestRegressor(n_estimators=100, random_state=42)
    production_model.fit(X, data['UnitsProduced'])
    models['production'] = production_model
    
    # Save models and scaler
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    return models

def train_quality_model(data):
    """Train product quality prediction model"""
    # Feature engineering
    features = ['Temperature', 'Vibration', 'Pressure', 'OperatingHours']
    X = data[features]
    
    # Create quality classes based on thresholds
    y = pd.cut(data['ProductQuality'], 
               bins=[-np.inf, 0.5, 0.7, 0.9, np.inf],
               labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'quality_model.joblib'))
    return model

def train_production_model(data):
    """Train production forecasting model"""
    # Group by day and machine
    daily_data = data.groupby(['MachineID', pd.Grouper(key='Timestamp', freq='D')])\
        .agg({'UnitsProduced':'sum', 'Temperature':'mean', 'Vibration':'mean', 'Pressure':'mean'})\
        .reset_index()
    
    # Create lag features
    for lag in [1, 2, 3, 7]:
        daily_data[f'Units_lag_{lag}'] = daily_data.groupby('MachineID')['UnitsProduced'].shift(lag)
    
    daily_data.dropna(inplace=True)
    
    # Train model
    features = ['Temperature', 'Vibration', 'Pressure'] + [f'Units_lag_{lag}' for lag in [1, 2, 3, 7]]
    X = daily_data[features]
    y = daily_data['UnitsProduced']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'production_model.joblib'))
    return model

def predict_maintenance(machine_data, models):
    """Predict maintenance needs for next 24 hours"""
    latest = machine_data.groupby('MachineID').last().reset_index()
    
    # Feature engineering
    latest['TimeSinceMaintenance'] = latest.groupby('MachineID')['MaintenanceFlag']\
        .cumsum().groupby(latest['MachineID']).cumcount()
    
    features = ['Temperature', 'Vibration', 'Pressure', 'OperatingHours', 'TimeSinceMaintenance']
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    X = scaler.transform(latest[features])
    
    predictions = []
    for hours in range(1, 25):
        pred = latest.copy()
        pred['Timestamp'] = pred['Timestamp'] + timedelta(hours=hours)
        
        # Predict sensor values
        for param in ['Temperature', 'Vibration', 'Pressure']:
            model = models[f'maintenance_{param}']
            pred[param] = model.predict(X)
            X[:, features.index(param)] = pred[param]
        
        # Predict maintenance probability
        maint_prob = models['maintenance_classifier'].predict_proba(X)[:,1]
        pred['MaintenanceProbability'] = maint_prob
        
        pred['OperatingHours'] += 1
        pred['TimeSinceMaintenance'] += 1
        predictions.append(pred)
    
    return pd.concat(predictions)

def predict_quality(machine_data, models):
    """Predict product quality for next 24 hours"""
    latest = machine_data.groupby('MachineID').last().reset_index()
    
    # Feature engineering
    latest['TimeSinceMaintenance'] = latest.groupby('MachineID')['MaintenanceFlag']\
        .cumsum().groupby(latest['MachineID']).cumcount()
    
    features = ['Temperature', 'Vibration', 'Pressure', 'OperatingHours', 'TimeSinceMaintenance']
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    X = scaler.transform(latest[features])
    
    predictions = []
    for hours in range(1, 25):
        pred = latest.copy()
        pred['Timestamp'] = pred['Timestamp'] + timedelta(hours=hours)
        
        # Predict quality probability - handle the case where predict_proba might return only one class
        quality_proba = models['quality'].predict_proba(X)
        if quality_proba.shape[1] > 1:
            # If we have probabilities for multiple classes, take the positive class (index 1)
            pred['QualityProbability'] = quality_proba[:, 1]
        else:
            # If we only have one class, use the direct prediction (0 or 1)
            pred['QualityProbability'] = models['quality'].predict(X)
        
        # Update features for next prediction
        for param in ['Temperature', 'Vibration', 'Pressure']:
            model = models[f'maintenance_{param}']
            pred[param] = model.predict(X)
            X[:, features.index(param)] = pred[param]
        
        pred['OperatingHours'] += 1
        pred['TimeSinceMaintenance'] += 1
        predictions.append(pred)
    
    return pd.concat(predictions)

def predict_production(machine_data, models):
    """Predict production output for next 24 hours"""
    latest = machine_data.groupby('MachineID').last().reset_index()
    
    # Feature engineering
    latest['TimeSinceMaintenance'] = latest.groupby('MachineID')['MaintenanceFlag']\
        .cumsum().groupby(latest['MachineID']).cumcount()
    
    features = ['Temperature', 'Vibration', 'Pressure', 'OperatingHours', 'TimeSinceMaintenance']
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    X = scaler.transform(latest[features])
    
    predictions = []
    for hours in range(1, 25):
        pred = latest.copy()
        pred['Timestamp'] = pred['Timestamp'] + timedelta(hours=hours)
        
        # Predict production output
        pred['UnitsProduced'] = models['production'].predict(X)
        
        # Update features for next prediction
        for param in ['Temperature', 'Vibration', 'Pressure']:
            model = models[f'maintenance_{param}']
            pred[param] = model.predict(X)
            X[:, features.index(param)] = pred[param]
        
        pred['OperatingHours'] += 1
        pred['TimeSinceMaintenance'] += 1
        predictions.append(pred)
    
    return pd.concat(predictions)

def show_maintenance_tab(machine_data, models):
    """Display maintenance predictions"""
    st.header("🛠️ Maintenance Predictions")
    
    # Machine selection
    machines = machine_data['MachineID'].unique()
    selected_machine = st.selectbox("Select Machine", machines, key="maint_machine")
    
    # Filter data
    machine_subset = machine_data[machine_data['MachineID'] == selected_machine].copy()
    
    # Get predictions
    predictions = predict_maintenance(machine_subset, models)
    
    # Create visualization
    fig = go.Figure()
    
    # Plot maintenance probability
    fig.add_trace(go.Scatter(
        x=predictions['Timestamp'],
        y=predictions['MaintenanceProbability'],
        mode='lines+markers',
        name='Maintenance Probability',
        line=dict(width=2),
        yaxis='y'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="Maintenance Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Maintenance Probability for {selected_machine}",
        yaxis_title="Probability",
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show maintenance recommendations
    st.subheader("Maintenance Recommendations")
    latest_pred = predictions.iloc[-1]
    
    if latest_pred['MaintenanceProbability'] > 0.7:
        st.error(f"🚨 Immediate maintenance recommended for {selected_machine} (Probability: {latest_pred['MaintenanceProbability']:.2%})")
    elif latest_pred['MaintenanceProbability'] > 0.5:
        st.warning(f"⚠️ Maintenance suggested for {selected_machine} soon (Probability: {latest_pred['MaintenanceProbability']:.2%})")
    else:
        st.success(f"✅ {selected_machine} is operating normally (Probability: {latest_pred['MaintenanceProbability']:.2%})")

def show_quality_tab(machine_data, models):
    """Display quality predictions"""
    st.header("🏆 Quality Predictions")
    
    # Machine selection
    machines = machine_data['MachineID'].unique()
    selected_machine = st.selectbox("Select Machine", machines, key="quality_machine")
    
    # Filter data
    machine_subset = machine_data[machine_data['MachineID'] == selected_machine].copy()
    
    # Get predictions
    predictions = predict_quality(machine_subset, models)
    
    # Create visualization
    fig = go.Figure()
    
    # Plot quality probability
    fig.add_trace(go.Scatter(
        x=predictions['Timestamp'],
        y=predictions['QualityProbability'],
        mode='lines+markers',
        name='Quality Probability',
        line=dict(width=2),
        yaxis='y'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=QUALITY_THRESHOLDS['Good'],
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="Quality Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Quality Probability for {selected_machine}",
        yaxis_title="Probability",
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show quality recommendations
    st.subheader("Quality Recommendations")
    latest_pred = predictions.iloc[-1]
    
    if latest_pred['QualityProbability'] < QUALITY_THRESHOLDS['Good']:
        st.error(f"❌ Quality issues predicted for {selected_machine} (Probability: {latest_pred['QualityProbability']:.2%})")
    else:
        st.success(f"✅ {selected_machine} is producing good quality (Probability: {latest_pred['QualityProbability']:.2%})")

def show_production_tab(machine_data, models):
    """Display production predictions"""
    st.header("📦 Production Predictions")
    
    # Machine selection
    machines = machine_data['MachineID'].unique()
    selected_machine = st.selectbox("Select Machine", machines, key="prod_machine")
    
    # Filter data
    machine_subset = machine_data[machine_data['MachineID'] == selected_machine].copy()
    
    # Get predictions
    predictions = predict_production(machine_subset, models)
    
    # Create visualization
    fig = go.Figure()
    
    # Plot production output
    fig.add_trace(go.Scatter(
        x=predictions['Timestamp'],
        y=predictions['UnitsProduced'],
        mode='lines+markers',
        name='Production Output',
        line=dict(width=2),
        yaxis='y'
    ))
    
    # Add target line
    fig.add_hline(
        y=PRODUCTION_TARGET,
        line_dash="dash",
        line_color="green",
        opacity=0.7,
        annotation_text="Production Target",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Production Output for {selected_machine}",
        yaxis_title="Units Produced",
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show production recommendations
    st.subheader("Production Recommendations")
    latest_pred = predictions.iloc[-1]
    
    if latest_pred['UnitsProduced'] < PRODUCTION_TARGET * 0.9:
        st.error(f"⚠️ Production below target for {selected_machine} ({latest_pred['UnitsProduced']:.1f} units vs target {PRODUCTION_TARGET})")
    else:
        st.success(f"✅ {selected_machine} is meeting production targets ({latest_pred['UnitsProduced']:.1f} units)")

def main():
    """Main application with three tabs"""
    st.title("Factory Predictive System")
    
    # Data loading
    st.sidebar.header("Data Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload machine data (CSV)", type="csv")
    
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            if not all(col in new_data.columns for col in REQUIRED_COLUMNS):
                st.error("Uploaded file missing required columns")
            else:
                save_path = os.path.join(DATA_DIR, uploaded_file.name)
                new_data.to_csv(save_path, index=False)
                st.success(f"Data saved to {save_path}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Load data
    machine_data = load_real_data()
    if machine_data is None:
        return
    
    # Model training
    models = {}
    if st.sidebar.button("Train Models") or any(f.endswith('.joblib') for f in os.listdir(MODEL_DIR)):
        with st.spinner("Loading models..."):
            try:
                models = {
                    'maintenance_Temperature': joblib.load(os.path.join(MODEL_DIR, 'maintenance_Temperature_model.joblib')),
                    'maintenance_Vibration': joblib.load(os.path.join(MODEL_DIR, 'maintenance_Vibration_model.joblib')),
                    'maintenance_Pressure': joblib.load(os.path.join(MODEL_DIR, 'maintenance_Pressure_model.joblib')),
                    'maintenance_classifier': joblib.load(os.path.join(MODEL_DIR, 'maintenance_classifier_model.joblib')),
                    'quality': joblib.load(os.path.join(MODEL_DIR, 'quality_model.joblib')),
                    'production': joblib.load(os.path.join(MODEL_DIR, 'production_model.joblib'))
                }
                st.success("Models loaded successfully!")
            except:
                with st.spinner("Training new models..."):
                    models = train_models(machine_data)
                    st.success("Models trained successfully!")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Machine Maintenance", "Quality Prediction", "Production Forecast"])
    
    with tab1:
        if models:
            show_maintenance_tab(machine_data, models)
        else:
            st.warning("Please train models first")
    
    with tab2:
        if models:
            show_quality_tab(machine_data, models)
        else:
            st.warning("Please train models first")
    
    with tab3:
        if models:
            show_production_tab(machine_data, models)
        else:
            st.warning("Please train models first")

if __name__ == "__main__":
    main()