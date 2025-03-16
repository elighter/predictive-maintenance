# 🔧 Industrial Maintenance Prediction Project with Machine Learning

This project aims to predict industrial machine failures in advance using machine learning techniques. It provides a comprehensive framework for developing predictive maintenance strategies using sensor data.

## 📋 Project Contents

The project includes the following main components:

1. **Interactive Visualizations** (`interactive_visualizations.py`): Interactive analysis of the dataset using Plotly
2. **Model Interpretability** (`model_interpretability.py`): Model explanations with SHAP values and Partial Dependency Plots (PDP)
3. **Streamlit Dashboard** (`maintenance_dashboard.py`): Comprehensive, user-friendly analysis dashboard

## 🚀 Setup and Running

Follow these steps to run the project:

### 1. Install Required Libraries

Install the necessary libraries to run the project:

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

The dataset used in this project is the [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) dataset from Kaggle.

There are two methods to download the dataset:

#### a) Direct Download
You can download the dataset directly using the following command:

```bash
mkdir -p data
curl -L -o ~/Downloads/machine-predictive-maintenance-classification.zip \
  https://www.kaggle.com/api/v1/datasets/download/shivamb/machine-predictive-maintenance-classification
unzip ~/Downloads/machine-predictive-maintenance-classification.zip -d ./data
```

#### b) Manual Download
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
2. Click the "Download" button to download the dataset
3. Extract the ZIP file
4. Copy the CSV file to the `data` folder in the project root directory (create the `data` folder if needed)

### 3. Run Project Components

#### a) Interactive Visualizations

To create interactive Plotly graphs and save them as HTML files:

```bash
python interactive_visualizations.py
```

This command will create the following HTML files:
- `failure_distribution.html`: Failure distribution pie chart
- `failure_types.html`: Failure types distribution bar chart
- `3d_sensor_space.html`: 3D sensor data distribution chart
- `correlation_matrix.html`: Interactive correlation matrix
- `product_types_failure.html`: Failure rates by product type
- `sensor_distribution.html`: Sensor values distribution by failure status
- `temperature_torque_animation.html`: Animated change graph based on tool wear

#### b) Model Interpretability

To create SHAP and PDP graphs:

```bash
python model_interpretability.py
```

This command will create various model interpretation graphs in PNG format:
- Confusion matrices
- Feature importance levels
- SHAP importance graphs
- SHAP summary graphs
- SHAP decision graphs
- Partial dependency plots

#### c) Streamlit Dashboard

To run the interactive dashboard:

```bash
streamlit run maintenance_dashboard.py
```

This command will launch a web application with the following features:
- Data analysis visualizations
- Dynamic filters
- Model training and evaluation
- Failure prediction simulation based on sensor values

## 📊 Features

### Data Analysis
- Multivariate sensor data correlation analysis
- Examination of failure distribution and types
- Product type and failure relationship analysis
- Machine behavior visualization in 3D sensor space

### Modeling
- Random Forest and Gradient Boosting classifiers
- Cross-validation and hyperparameter optimization
- Performance metrics and evaluation graphs

### Model Interpretability
- Feature importance with SHAP values
- Individual prediction explanations with SHAP summary graphs
- Feature effects with Partial Dependency Plots (PDP)

### Dashboard
- Interactive data exploration
- Real-time model training
- User input prediction simulation
- Filtering and feature selection

## 📈 Results

This project is an effective example of applying machine learning for developing predictive maintenance strategies for industrial machines. The results show that:

- Machine failures can be predicted with high accuracy
- Failure probability increases especially in type H machines and high tool wear conditions
- Temperature difference and rotation speed are critical monitoring parameters
- Resources can be used more efficiently with data-driven maintenance decisions

## 🏗️ Project Architecture Flow Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│                 │     │                     │     │                      │
│  Data Collection│────▶│  Data Preprocessing │────▶│  Exploratory Data    │
│  (Kaggle)       │     │                     │     │  Analysis (viz.py)   │
│                 │     │                     │     │                      │
└─────────────────┘     └─────────────────────┘     └──────────┬───────────┘
                                                               │
                                                               ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│                 │     │                     │     │                      │
│  Prediction     │◀────│  Model Evaluation   │◀────│  Model Training      │
│  Service        │     │  (interpretability) │     │                      │
│                 │     │                     │     │                      │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
        │                                                      ▲
        │                                                      │
        ▼                                                      │
┌─────────────────┐                                  ┌──────────────────────┐
│                 │                                  │                      │
│  Streamlit      │──────────────────────────────────▶  Dynamic Model       │
│  Dashboard      │                                  │  Training & Analysis │
│                 │                                  │                      │
└─────────────────┘                                  └──────────────────────┘
```

## 💻 System Requirements

- Python 3.7 or higher
- 4GB RAM (minimum)
- Modern web browser for Streamlit dashboard (Chrome, Firefox, Edge, etc.)
- Operating System: Windows, MacOS, or Linux

## 🧰 Development

To contribute to the project:

1. Fork it
2. Create a new feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Added new feature'`)
4. Push your branch (`git push origin feature/new-feature`)
5. Create a pull request

## 📝 License

This project is distributed under the [MIT License](LICENSE).

## 📧 Contact

For questions and suggestions: emrecakmak@me.com

---

#### Sources and References

1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
2. SHAP: A Game Theoretic Approach to Explain the Output of any Machine Learning Model, Lundberg & Lee, NIPS 2017.
3. Kaggle Dataset: "Machine Predictive Maintenance Classification", Shivam Bansal, 2022.
4. McKinsey & Company. "Predictive maintenance: Taking proactive measures based on advanced data analytics to predict and avoid machine failure."

## 📊 Basic Workflow

The project follows the basic machine learning workflow below:

### 1. Data Loading and Preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/predictive_maintenance.csv")

# Separate features and target variable
X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
y = df['Target']

# Identify categorical and numerical variables
categorical_features = ['Type']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
```

### 2. Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)
```

### 3. Model Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Make predictions on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Performance metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

### 4. Feature Importance

```python
import numpy as np

# Get feature importance
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = model.named_steps['classifier'].feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

# Show most important features
plt.figure(figsize=(10, 8))
sns.barplot(x=importances[indices][:10], y=[feature_names[i] for i in indices][:10])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
```

### 5. Making Predictions

```python
# Create sample data
sample_data = pd.DataFrame({
    'Air temperature [K]': [300],
    'Process temperature [K]': [310],
    'Rotational speed [rpm]': [1500],
    'Torque [Nm]': [40],
    'Tool wear [min]': [100],
    'Type': ['M']
})

# Make prediction
prediction = model.predict(sample_data)[0]
prediction_proba = model.predict_proba(sample_data)[0][1]

print(f"Prediction: {'Failure' if prediction == 1 else 'Normal'}")
print(f"Failure Probability: {prediction_proba:.4f}")
```

## ⚙️ Dataset Structure

The dataset used contains the following features:

- **UDI**: Equipment identifier
- **Product ID**: Product identifier
- **Type**: Machine type (L = Low, M = Medium, H = High)
- **Air temperature [K]**: Air temperature (Kelvin)
- **Process temperature [K]**: Process temperature (Kelvin)
- **Rotational speed [rpm]**: Rotational speed (RPM)
- **Torque [Nm]**: Torque value (Nm)
- **Tool wear [min]**: Tool wear (minutes)
- **Target**: Failure status (0 = Normal, 1 = Failure)
- **Failure Type**: Type of failure (Only specified in case of failure)

### Model Selection and Hyperparameters

The Random Forest classifier model was chosen for these advantages:

1. High accuracy and generalization capability
2. Tendency to reduce overfitting
3. Ability to determine feature importance
4. Ability to process categorical and numerical data together

Hyperparameters:
- **n_estimators=100**: Using 100 decision trees
- **random_state=42**: Fixed seed value for reproducibility

## 🏭 System Architecture for Real-Time Production Environment

When implementing this project in a real-time production environment, the following layered reference architecture can be used:

### Production Environment Reference Architecture

```
┌───────────────────────────── ManufactureIQ - AI-Powered Production Optimization Platform ─────────────────────────────┐
│                                                                                                                      │
│ ┌──────────────────────────────────────────── Edge Layer ────────────────────────────────────────────────┐          │
│ │                                                                                                         │          │
│ │    ┌──────────┐      ┌─────────┐      ┌────────┐      ┌─────┐      ┌────────┐      ┌──────────┐       │          │
│ │    │    IoT   │      │  Edge   │      │        │      │     │      │        │      │          │       │          │
│ │    │ Sensors  │      │   AI    │      │ SCADA  │      │ MES │      │PLC/RTU │      │ Cameras  │       │          │
│ │    └──────────┘      └─────────┘      └────────┘      └─────┘      └────────┘      └──────────┘       │          │
│ │                                                                                                         │          │
│ └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘          │
│                                                      │                                                                │
│ ┌────────────────────────────────── Data Integration Layer ──────────────────────────────────────────────┐          │
│ │                                                                                                         │          │
│ │    ┌──────────┐      ┌─────────┐      ┌────────────────┐      ┌───────────────────────┐               │          │
│ │    │  Apache  │      │         │      │   Kong API     │      │        gRPC           │               │          │
│ │    │  Kafka   │      │  Dapr   │      │   Gateway      │      │ Microservice Comm.    │               │          │
│ │    └──────────┘      └─────────┘      └────────────────┘      └───────────────────────┘               │          │
│ │                                                                                                         │          │
│ └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘          │
│                                                      │                                                                │
│ ┌──────────────────────────────────── Processing Layer ──────────────────────────────────────────────────┐          │
│ │                                                                                                         │          │
│ │    ┌────────────────┐      ┌─────────────┐      ┌─────────────────┐      ┌─────────────┐               │          │
│ │    │   InfluxDB     │      │    Redis    │      │    Kubeflow     │      │             │               │          │
│ │    │ (Time-series)  │      │  (Caching)  │      │  (ML Pipeline)  │      │ TimescaleDB │               │          │
│ │    └────────────────┘      └─────────────┘      └─────────────────┘      └─────────────┘               │          │
│ │                                                                                                         │          │
│ └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘          │
│                                                      │                                                                │
│ ┌───────────────────────────────────── Application Layer ──────────────────────────────────────────────┐           │
│ │                                                                                                       │           │
│ │  ┌──────────────┐    ┌────────────────┐    ┌────────────────────┐    ┌──────────────┐    ┌─────┐     │           │
│ │  │   Anomaly    │    │   Predictive   │    │      Energy        │    │    Quality   │    │     │     │           │
│ │  │  Detection   │    │  Maintenance   │    │   Optimization     │    │   Control    │    │ API │     │           │
│ │  └──────────────┘    └────────────────┘    └────────────────────┘    └──────────────┘    └─────┘     │           │
│ │                                                                                                       │           │
│ └───────────────────────────────────────────────────────────────────────────────────────────────────────┘           │
│                                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
