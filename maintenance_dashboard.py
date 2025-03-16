#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Industrial Maintenance Prediction Project with Machine Learning
Streamlit Dashboard Application

This module provides an interactive analytics dashboard using Streamlit
to analyze machine maintenance datasets and visualize 
predictive model results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style definitions
st.markdown("""
<style>
    /* Global styles to override Streamlit's defaults */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Modify Streamlit widgets background */
    .stTextInput > div > div {
        background-color: #262730;
    }
    
    .stSelectbox > div > div {
        background-color: #262730;
    }
    
    .stMultiSelect > div > div {
        background-color: #262730;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(25, 39, 52, 0.8);
        border-bottom: 3px solid #4e73df;
    }
    
    /* Sub-headers */
    .sub-header {
        font-size: 1.8rem;
        color: #ffffff;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(25, 39, 52, 0.8);
        border-left: 5px solid #4e73df;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(25, 39, 52, 0.8);
        border-left: 0.5rem solid #3498db;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: rgba(25, 39, 52, 0.8) !important;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(78, 115, 223, 0.5);
        color: #ffffff !important;
    }
    
    /* Metric labels */
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Metric values */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Tab sub-headers */
    .tab-subheader {
        font-size: 1.4rem;
        color: #ffffff;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        padding: 8px;
        border-radius: 6px;
        background-color: rgba(25, 39, 52, 0.8);
        border-left: 4px solid #4e73df;
    }
    
    /* Tab styles */
    button[data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 600;
        background-color: rgba(25, 39, 52, 0.6);
        border-radius: 5px 5px 0 0;
        padding: 5px 15px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background-color: rgba(78, 115, 223, 0.5);
        border-bottom: 3px solid #4e73df;
    }
    
    /* Chart containers */
    .stPlotlyChart {
        background-color: rgba(25, 39, 52, 0.7);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px !important;
        border: 1px solid rgba(78, 115, 223, 0.3);
    }
    
    /* DataTable styling */
    .dataframe-container {
        background-color: rgba(25, 39, 52, 0.7);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid rgba(78, 115, 223, 0.3);
    }
    
    .dataframe {
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #2c3e50;
        font-size: 0.8rem;
        color: #cccccc;
        background-color: rgba(25, 39, 52, 0.5);
        border-radius: 0 0 10px 10px;
    }
    
    /* Streamlit sidebar */
    .css-sidebar {
        background-color: #192734 !important;
    }
    
    /* Sidebar headers */
    .css-sidebar .css-header {
        background-color: #192734 !important;
    }
    
    /* Dataframes and tables */
    .stDataFrame {
        background-color: rgba(25, 39, 52, 0.7) !important;
    }
    
    div[data-testid="stTable"] {
        background-color: rgba(25, 39, 52, 0.7) !important;
    }
    
    /* Table cells */
    div[data-testid="stTable"] th {
        background-color: rgba(37, 51, 74, 0.7) !important;
        color: white !important;
    }
    
    div[data-testid="stTable"] td {
        background-color: rgba(25, 39, 52, 0.5) !important;
        color: white !important;
    }
    
    /* Disable Streamlit default background */
    .css-18e3th9 {
        background-color: transparent !important;
    }
    
    .css-1d391kg {
        background-color: #192734 !important;
    }
    
    /* Text and headers */
    p, h1, h2, h3, h4, h5, h6, li {
        color: #ffffff !important;
    }
    
    /* Success/Error/Warning/Info boxes */
    div.stAlert > div {
        background-color: rgba(25, 39, 52, 0.7) !important;
        color: white !important;
    }
    
    /* Success box */
    div.stAlert.success > div {
        border-left-color: #2ECC71 !important;
    }
    
    /* Warning box */
    div.stAlert.warning > div {
        border-left-color: #F1C40F !important;
    }
    
    /* Error box */
    div.stAlert.error > div {
        border-left-color: #E74C3C !important;
    }
    
    /* Info box */
    div.stAlert.info > div {
        border-left-color: #3498DB !important;
    }
</style>
""", unsafe_allow_html=True)

# Modify all Plotly charts to use dark theme by default
layout_template = dict(
    layout=dict(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
)

@st.cache_data
def load_data():
    """Loads and caches the maintenance dataset from local file"""
    try:
        file_path = "data/predictive_maintenance.csv"
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_data(df):
    """Processes and preprocesses the data"""
    # Separate target variable and features
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    y = df['Target']
    
    # Separate categorical and numerical features
    categorical_features = ['Type']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # Data preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

@st.cache_resource
def train_model(X_train, y_train, _preprocessor, model_name="Random Forest"):
    """Train the selected model"""
    # Create pipeline
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', _preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance"""
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Performance metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Data for ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return metrics, cm, fpr, tpr, roc_auc

def plot_failure_distribution(df):
    """Failure distribution pie chart"""
    failure_counts = df['Target'].value_counts().reset_index()
    failure_counts.columns = ['Status', 'Count']
    failure_counts['Status'] = failure_counts['Status'].map({0: 'Normal', 1: 'Failure'})
    
    fig = px.pie(
        failure_counts, 
        names='Status',
        values='Count',
        hole=0.4,
        title='Machine Status Distribution',
        color='Status',
        color_discrete_map={'Normal': '#2ECC71', 'Failure': '#E74C3C'},
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label', 
                     textfont_size=14, pull=[0, 0.1])
    
    # Apply dark theme
    fig.update_layout(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig

def plot_failure_types(df):
    """Failure type distribution"""
    # Failure type distribution
    failure_types = df[df['Target'] == 1]['Failure Type'].value_counts().reset_index()
    failure_types.columns = ['Failure Type', 'Count']
    
    # Define color scale
    colors = px.colors.qualitative.Plotly
    
    # Bar chart
    fig = px.bar(
        failure_types, 
        x='Failure Type', 
        y='Count',
        color='Failure Type',
        text='Count',
        title='Failure Type Distribution',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='outside', 
        textfont=dict(size=14),
        marker_line_width=2
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title='Failure Type',
        yaxis_title='Count',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=600,
        width=800,
        showlegend=False,
        # Dark theme
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_failure_by_type(df):
    """Failure rates by product type"""
    type_failure = df.groupby('Type')['Target'].mean().reset_index()
    type_failure.columns = ['Product Type', 'Failure Rate']
    type_failure['Failure Rate'] = type_failure['Failure Rate'] * 100
    
    # Product type descriptions
    type_names = {
        'L': 'Low',
        'M': 'Medium',
        'H': 'High'
    }
    
    type_failure['Type Description'] = type_failure['Product Type'].map(type_names)
    
    fig = px.bar(
        type_failure, 
        x='Product Type', 
        y='Failure Rate',
        color='Product Type',
        text='Failure Rate',
        hover_data=['Type Description'],
        title='Failure Rates by Product Type (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    
    # Dark theme
    fig.update_layout(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_correlation_heatmap(df):
    """Correlation matrix heatmap"""
    # Select numerical variables
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 
                  'Tool wear [min]', 'Target']
    
    # Calculate correlation matrix
    corr = df[numeric_cols].corr().round(2)
    
    # Heatmap visualization
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Correlation Matrix Between Variables'
    )
    
    # Dark theme
    fig.update_layout(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_3d_sensor_space(df):
    """3D sensor space failure distribution"""
    fig = px.scatter_3d(
        df,
        x='Air temperature [K]',
        y='Process temperature [K]',
        z='Rotational speed [rpm]',
        color='Target',
        size='Tool wear [min]',
        opacity=0.7,
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        symbol='Type',
        title='3D Sensor Space Failure Distribution',
        labels={
            'Target': 'Failure Status',
            'Type': 'Product Type'
        }
    )
    
    # Dark theme
    fig.update_layout(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        scene=dict(
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)', backgroundcolor='rgba(25, 39, 52, 0.7)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)', backgroundcolor='rgba(25, 39, 52, 0.7)'),
            zaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)', backgroundcolor='rgba(25, 39, 52, 0.7)')
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_confusion_matrix(cm):
    """Confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(6, 5))
    # Set the figure background to dark
    fig.patch.set_facecolor('#192734')
    ax.set_facecolor('#192734')
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar=True
    )
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.set_title('Confusion Matrix', color='white')
    ax.set_xticklabels(['Normal', 'Failure'], color='white')
    ax.set_yticklabels(['Normal', 'Failure'], color='white')
    ax.tick_params(colors='white')
    
    # Set the color of the outline
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """ROC curve visualization"""
    fig, ax = plt.subplots(figsize=(6, 5))
    # Set the figure background to dark
    fig.patch.set_facecolor('#192734')
    ax.set_facecolor('#192734')
    
    ax.plot(fpr, tpr, color='#4e73df', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curve', color='white')
    ax.legend(loc="lower right")
    
    # Set the color of text in the legend
    legend = ax.get_legend()
    for text in legend.get_texts():
        text.set_color('white')
    
    # Set the tick colors
    ax.tick_params(colors='white')
    
    # Set the color of the outline
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        
    return fig

def plot_sensor_data_by_failure(df, sensor_name):
    """Sensor data distribution by failure status"""
    fig = px.histogram(
        df, 
        x=sensor_name, 
        color='Target',
        barmode='overlay',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        labels={'Target': 'Failure Status'},
        title=f'{sensor_name} Distribution'
    )
    
    fig.update_layout(bargap=0.1)
    
    # Dark theme
    fig.update_layout(
        paper_bgcolor='rgba(25, 39, 52, 0.7)',
        plot_bgcolor='rgba(25, 39, 52, 0.7)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def main():
    """Main Streamlit application"""
    # Title and introduction
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    This analytics dashboard provides visualizations of machine learning model results
    for machine maintenance datasets. Sensor data and product features are used to predict
    failure and perform performance analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Sidebar - Filtering options
        st.sidebar.markdown("## 🔍 Data Filtering")
        
        # Product type filtering
        product_types = sorted(df['Type'].unique())
        selected_types = st.sidebar.multiselect(
            "Select Product Type:",
            options=product_types,
            default=product_types
        )
        
        # Tool wear filtering
        min_tool_wear = int(df['Tool wear [min]'].min())
        max_tool_wear = int(df['Tool wear [min]'].max())
        tool_wear_range = st.sidebar.slider(
            "Tool Wear Range (min):",
            min_value=min_tool_wear,
            max_value=max_tool_wear,
            value=(min_tool_wear, max_tool_wear)
        )
        
        # Temperature range filtering
        min_temp = float(df['Air temperature [K]'].min())
        max_temp = float(df['Air temperature [K]'].max())
        temp_range = st.sidebar.slider(
            "Air Temperature Range (K):",
            min_value=min_temp,
            max_value=max_temp,
            value=(min_temp, max_temp)
        )
        
        # Apply filtering
        filtered_df = df[
            (df['Type'].isin(selected_types)) &
            (df['Tool wear [min]'] >= tool_wear_range[0]) &
            (df['Tool wear [min]'] <= tool_wear_range[1]) &
            (df['Air temperature [K]'] >= temp_range[0]) &
            (df['Air temperature [K]'] <= temp_range[1])
        ]
        
        # Sidebar - Model options
        st.sidebar.markdown("## 🧠 Model Settings")
        model_name = st.sidebar.selectbox(
            "Select Model:",
            ["Random Forest", "Gradient Boosting"]
        )
        
        # Sidebar - Prediction simulation
        st.sidebar.markdown("## 💭 Failure Prediction Simulation")
        
        # User input values
        air_temp = st.sidebar.number_input("Air Temperature (K)", min_value=float(df['Air temperature [K]'].min()), 
                                           max_value=float(df['Air temperature [K]'].max()), 
                                           value=float(df['Air temperature [K]'].mean()))
        
        process_temp = st.sidebar.number_input("Process Temperature (K)", min_value=float(df['Process temperature [K]'].min()), 
                                               max_value=float(df['Process temperature [K]'].max()), 
                                               value=float(df['Process temperature [K]'].mean()))
        
        rotation = st.sidebar.number_input("Rotational Speed (rpm)", min_value=float(df['Rotational speed [rpm]'].min()), 
                                           max_value=float(df['Rotational speed [rpm]'].max()), 
                                           value=float(df['Rotational speed [rpm]'].mean()))
        
        torque = st.sidebar.number_input("Torque (Nm)", min_value=float(df['Torque [Nm]'].min()), 
                                         max_value=float(df['Torque [Nm]'].max()), 
                                         value=float(df['Torque [Nm]'].mean()))
        
        tool_wear = st.sidebar.number_input("Tool Wear (min)", min_value=float(df['Tool wear [min]'].min()), 
                                            max_value=float(df['Tool wear [min]'].max()), 
                                            value=float(df['Tool wear [min]'].mean()))
        
        product_type = st.sidebar.selectbox("Product Type", options=df['Type'].unique())
        
        # Main tabs
        tabs = st.tabs(["📊 Data Analysis", "📈 Model Results", "🔮 Prediction Simulation"])
        
        with tabs[0]:
            st.markdown('<h2 class="sub-header">Data Analysis</h2>', unsafe_allow_html=True)
            
            # General data summary
            st.markdown('<h3 class="tab-subheader">Dataset Summary</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(filtered_df))
            with col2:
                st.metric("Failure Count", filtered_df['Target'].sum())
            with col3:
                failure_rate = filtered_df['Target'].mean() * 100
                st.metric("Failure Rate (%)", f"{failure_rate:.2f}%")
            
            # Data samples
            st.markdown('<h3 class="tab-subheader">Data Samples</h3>', unsafe_allow_html=True)
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistical summary
            st.markdown('<h3 class="tab-subheader">Statistical Summary</h3>', unsafe_allow_html=True)
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_df.describe().T, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown('<h3 class="tab-subheader">Visualizations</h3>', unsafe_allow_html=True)
            
            # 1st Row - Failure distribution and types
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_failure_distribution(filtered_df), use_container_width=True)
            with col2:
                if filtered_df['Target'].sum() > 0:
                    st.plotly_chart(plot_failure_types(filtered_df), use_container_width=True)
                else:
                    st.info("No failure data found with selected filters.")
            
            # 2nd Row - Product type and correlation
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_failure_by_type(filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)
            
            # 3rd Row - 3D visualization
            st.plotly_chart(plot_3d_sensor_space(filtered_df), use_container_width=True)
            
            # 4th Row - Sensor distributions
            sensor_tabs = st.tabs([
                "Air Temperature", 
                "Process Temperature", 
                "Rotational Speed", 
                "Torque", 
                "Tool Wear"
            ])
            
            with sensor_tabs[0]:
                st.plotly_chart(plot_sensor_data_by_failure(filtered_df, 'Air temperature [K]'), use_container_width=True)
            
            with sensor_tabs[1]:
                st.plotly_chart(plot_sensor_data_by_failure(filtered_df, 'Process temperature [K]'), use_container_width=True)
                
            with sensor_tabs[2]:
                st.plotly_chart(plot_sensor_data_by_failure(filtered_df, 'Rotational speed [rpm]'), use_container_width=True)
                
            with sensor_tabs[3]:
                st.plotly_chart(plot_sensor_data_by_failure(filtered_df, 'Torque [Nm]'), use_container_width=True)
                
            with sensor_tabs[4]:
                st.plotly_chart(plot_sensor_data_by_failure(filtered_df, 'Tool wear [min]'), use_container_width=True)
        
        with tabs[1]:
            st.markdown('<h2 class="sub-header">Model Results</h2>', unsafe_allow_html=True)
            
            # Model training and evaluation
            if st.button("Train and Evaluate Model"):
                with st.spinner("Training model..."):
                    # Prepare data
                    X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data(df)
                    
                    # Train model
                    pipeline = train_model(X_train, y_train, preprocessor, model_name)
                    
                    # Evaluate model
                    metrics, cm, fpr, tpr, roc_auc = evaluate_model(pipeline, X_test, y_test)
                    
                    # Performance metrics
                    st.markdown('<h3 class="tab-subheader">Model Performance Metrics</h3>', unsafe_allow_html=True)
                    metric_cols = st.columns(4)
                    for i, (metric, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            st.metric(metric, f"{value:.4f}")
                    
                    # Confusion matrix and ROC curve
                    st.markdown('<h3 class="tab-subheader">Model Evaluation Charts</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plot_confusion_matrix(cm))
                    with col2:
                        st.pyplot(plot_roc_curve(fpr, tpr, roc_auc))
                    
                    # Feature importance
                    st.markdown('<h3 class="tab-subheader">Feature Importance</h3>', unsafe_allow_html=True)
                    if hasattr(pipeline['classifier'], 'feature_importances_'):
                        # Get feature names
                        feature_names = numerical_features + ['Type_M', 'Type_H']
                        importances = pipeline['classifier'].feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        # Bar plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Set dark background
                        fig.patch.set_facecolor('#192734')
                        ax.set_facecolor('#192734')
                        
                        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax, palette='Blues_r')
                        ax.set_title('Feature Importance', color='white', fontsize=14)
                        ax.set_xlabel('Importance', color='white', fontsize=12)
                        ax.set_ylabel('Features', color='white', fontsize=12)
                        
                        # Set the tick colors
                        ax.tick_params(colors='white')
                        
                        # Set the color of the outline
                        for spine in ax.spines.values():
                            spine.set_edgecolor('white')
                            
                        st.pyplot(fig)
                    else:
                        st.info("Feature importance is not available for the selected model.")
            else:
                st.info("Click 'Train and Evaluate Model' button to train and evaluate the model.")
        
        with tabs[2]:
            st.markdown('<h2 class="sub-header">Prediction Simulation</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            In this section, you can predict the failure probability of a new machine using the input values on the left panel.
            Adjust the sensor values and product type, then click the prediction button.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.sidebar.button("Make Failure Prediction"):
                with st.spinner("Making prediction..."):
                    # Prepare data
                    X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data(df)
                    
                    # Train model
                    pipeline = train_model(X_train, y_train, preprocessor, model_name)
                    
                    # New data for prediction
                    new_data = pd.DataFrame({
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotation],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear],
                        'Type': [product_type]
                    })
                    
                    # Make prediction
                    prediction_proba = pipeline.predict_proba(new_data)[0][1]
                    prediction = 1 if prediction_proba > 0.5 else 0
                    
                    # Show results
                    st.markdown('<h3 class="tab-subheader">Prediction Result</h3>', unsafe_allow_html=True)
                    
                    # Prediction indicators
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ Failure Risk Detected!")
                        else:
                            st.success("✅ Normal Operation Expected")
                    
                    with col2:
                        # Progress bar with failure probability
                        st.metric("Failure Probability", f"{prediction_proba:.2%}")
                        st.progress(float(prediction_proba))
                    
                    # Sensor value summary
                    st.markdown('<h3 class="tab-subheader">Input Parameters</h3>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Air Temperature", f"{air_temp:.2f} K")
                        st.metric("Process Temperature", f"{process_temp:.2f} K")
                    with col2:
                        st.metric("Rotational Speed", f"{rotation:.2f} rpm")
                        st.metric("Torque", f"{torque:.2f} Nm")
                    with col3:
                        st.metric("Tool Wear", f"{tool_wear:.2f} min")
                        st.metric("Product Type", product_type)
                    
                    # Failure probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Failure Probability (%)", 'font': {'size': 24, 'color': 'white'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                            'bar': {'color': "#4e73df"},
                            'bgcolor': "rgba(25, 39, 52, 0.5)",
                            'borderwidth': 2,
                            'bordercolor': "#2c3e50",
                            'steps': [
                                {'range': [0, 25], 'color': '#2ECC71'},
                                {'range': [25, 50], 'color': '#F1C40F'},
                                {'range': [50, 75], 'color': '#E67E22'},
                                {'range': [75, 100], 'color': '#E74C3C'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor = 'rgba(25, 39, 52, 0.8)',
                        plot_bgcolor = 'rgba(25, 39, 52, 0.8)',
                        font = {'color': 'white'},
                        margin = dict(t=30, b=30, l=30, r=30)
                    )
                        
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown('<h3 class="tab-subheader">Maintenance Recommendations</h3>', unsafe_allow_html=True)
                    
                    # Recommendations based on failure status
                    if prediction_proba > 0.75:
                        st.error("""
                        🚨 **URGENT MAINTENANCE REQUIRED!**
                        - Stop the machine as soon as possible for detailed inspection
                        - Check tool wear and replace if necessary
                        - Monitor temperature values and check cooling system
                        """)
                    elif prediction_proba > 0.5:
                        st.warning("""
                        ⚠️ **UNPLANNED DOWNTIME RISK!**
                        - Schedule maintenance soon (within 1 week)
                        - Monitor tool condition closely
                        - Regularly check rotational speed and torque values
                        """)
                    elif prediction_proba > 0.25:
                        st.info("""
                        ℹ️ **LOW RISK**
                        - Continue with your normal maintenance program
                        - Monitor parameters regularly
                        - Consider tool replacement at next planned maintenance
                        """)
                    else:
                        st.success("""
                        ✅ **NORMAL OPERATION**
                        - Normal operating conditions can be maintained
                        - Follow routine maintenance schedule
                        - Record sensor data regularly
            """)
    else:
        st.info("Adjust the values in the left panel and click 'Make Failure Prediction' to make a prediction.")
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("Industrial Maintenance Prediction Project © 2024")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 