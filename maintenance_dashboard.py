#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi 
Streamlit Dashboard Uygulaması

Bu modül, makine bakımı veri setini analiz etmek ve 
prediktif model sonuçlarını görselleştirmek için Streamlit kullanarak
interaktif bir analiz panosu sunar.
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

# Makine öğrenmesi kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Sayfa yapılandırması
st.set_page_config(
    page_title="Prediktif Bakım Analiz Panosu",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stil tanımlamaları
st.markdown("""
<style>
    /* Ana başlıklar */
    .main-header {
        font-size: 2.8rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 15px;
        border-radius: 10px;
        border-bottom: 3px solid #4e73df;
    }
    
    /* Alt başlıklar */
    .sub-header {
        font-size: 1.8rem;
        color: #ffffff;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #4e73df;
    }
    
    /* Bilgi kutusu */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(25, 39, 52, 0.8);
        border-left: 0.5rem solid #3498db;
        margin-bottom: 1rem;
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Metrik kartları */
    .stMetric {
        background-color: rgba(25, 39, 52, 0.8);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(78, 115, 223, 0.5);
        color: #ffffff;
    }
    
    /* Metrik kartlarındaki label */
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Metrik kartlarındaki değer */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }
    
    /* Alt sayfa başlıkları */
    .tab-subheader {
        font-size: 1.4rem;
        color: #ffffff;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        padding: 8px;
        border-radius: 6px;
        border-left: 4px solid #4e73df;
    }
    
    /* Sekme stilleri */
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
    
    /* Grafik kartları */
    .stPlotlyChart {
        background-color: rgba(25, 39, 52, 0.7);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin-bottom: 20px !important;
        border: 1px solid rgba(78, 115, 223, 0.3);
    }
    
    /* Tablo stilleri */
    .dataframe-container {
        background-color: rgba(25, 39, 52, 0.7);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        border: 1px solid rgba(78, 115, 223, 0.3);
    }
    
    .dataframe {
        color: #ffffff !important;
    }
    
    div[data-testid="stTable"] {
        color: #ffffff !important;
    }
    
    /* Yan panel başlıkları */
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Yan panel metin elemanları */
    div[data-testid="stSidebar"] div[data-testid="stText"] {
        color: #ffffff !important;
    }
    
    /* Kontrol paneli stileri */
    .sidebar .sidebar-content {
        background-color: rgba(25, 39, 52, 0.9);
    }
    
    /* Metin renkleri */
    p, label, li {
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid rgba(78, 115, 223, 0.3);
        font-size: 0.8rem;
        color: #ffffff;
    }
    
    /* Streamlit öğeleri düzenlemeleri */
    div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    
    /* Tüm renk düzenlemeleri */
    .main .block-container {
        background-color: transparent !important;
    }
    
    /* Tablo arka plan renkleri */
    div[data-testid="stTable"] th {
        background-color: rgba(25, 39, 52, 0.9) !important;
        color: white !important;
    }
    
    div[data-testid="stTable"] td {
        background-color: rgba(25, 39, 52, 0.7) !important;
        color: white !important;
    }
    
    /* İçerik kutuları */
    .element-container {
        background-color: transparent !important;
    }
    
    /* Streamlit formları */
    .stForm {
        background-color: rgba(25, 39, 52, 0.7) !important;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(78, 115, 223, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Bakım veri setini yerel dosyadan yükler ve cache'ler"""
    try:
        file_path = "data/predictive_maintenance.csv"
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

@st.cache_data
def prepare_data(df):
    """Veriyi işler ve önişleme yapar"""
    # Hedef değişkeni ve özellikleri ayır
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    y = df['Target']
    
    # Kategorik ve sayısal değişkenleri ayır
    categorical_features = ['Type']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # Veri ön işleme pipeline'ı
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

@st.cache_resource
def train_model(X_train, y_train, preprocessor, model_name="Random Forest"):
    """Seçilen modeli eğit"""
    # Pipeline oluştur
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    
    # Pipeline'ı oluştur
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Modeli eğit
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Model performansını değerlendir"""
    # Tahmin yap
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Performans metrikleri
    metrics = {
        'Doğruluk': accuracy_score(y_test, y_pred),
        'Kesinlik': precision_score(y_test, y_pred),
        'Duyarlılık': recall_score(y_test, y_pred),
        'F1 Skoru': f1_score(y_test, y_pred)
    }
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC eğrisi için veriler
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return metrics, cm, fpr, tpr, roc_auc

def plot_failure_distribution(df):
    """Arıza dağılımı pasta grafiği"""
    failure_counts = df['Target'].value_counts().reset_index()
    failure_counts.columns = ['Durum', 'Sayı']
    failure_counts['Durum'] = failure_counts['Durum'].map({0: 'Normal', 1: 'Arıza'})
    
    fig = px.pie(
        failure_counts, 
        values='Sayı', 
        names='Durum',
        title='Makine Durumu Dağılımı',
        color='Durum',
        color_discrete_map={'Normal': '#2ECC71', 'Arıza': '#E74C3C'},
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label', 
                     textfont_size=14, pull=[0, 0.1])
    
    return fig

def plot_failure_types(df):
    """Arıza türleri dağılımı"""
    # Arıza türleri dağılımı
    failure_types = df[df['Target'] == 1]['Failure Type'].value_counts().reset_index()
    failure_types.columns = ['Arıza Türü', 'Sayı']
    
    # Renk skalasını tanımlayalım
    colors = px.colors.qualitative.Plotly
    
    # Bar grafiği
    fig = px.bar(
        failure_types, 
        x='Arıza Türü', 
        y='Sayı',
        color='Arıza Türü',
        text='Sayı',
        title='Arıza Türleri Dağılımı',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='outside', 
        textfont=dict(size=14),
        marker_line_width=2
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title='Arıza Türü',
        yaxis_title='Sayı',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=600,
        width=800,
        showlegend=False
    )
    
    return fig

def plot_failure_by_type(df):
    """Ürün tiplerine göre arıza oranları"""
    type_failure = df.groupby('Type')['Target'].mean().reset_index()
    type_failure.columns = ['Ürün Tipi', 'Arıza Oranı']
    type_failure['Arıza Oranı'] = type_failure['Arıza Oranı'] * 100
    
    # Ürün tipi açıklamaları
    type_names = {
        'L': 'Düşük',
        'M': 'Orta',
        'H': 'Yüksek'
    }
    
    type_failure['Tip Açıklaması'] = type_failure['Ürün Tipi'].map(type_names)
    
    fig = px.bar(
        type_failure, 
        x='Ürün Tipi', 
        y='Arıza Oranı',
        color='Ürün Tipi',
        text='Arıza Oranı',
        hover_data=['Tip Açıklaması'],
        title='Ürün Tiplerine Göre Arıza Oranları (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    
    return fig

def plot_correlation_heatmap(df):
    """Korelasyon matrisi ısı haritası"""
    # Sayısal değişkenleri seç
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 
                  'Tool wear [min]', 'Target']
    
    # Korelasyon matrisi hesapla
    corr = df[numeric_cols].corr().round(2)
    
    # Isı haritası görselleştirmesi
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Değişkenler Arası Korelasyon Matrisi'
    )
    
    return fig

def plot_3d_sensor_space(df):
    """3D sensör uzayında arıza dağılımı"""
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
        title='3D Sensör Uzayında Arıza Dağılımı',
        labels={
            'Target': 'Arıza Durumu',
            'Type': 'Ürün Tipi'
        }
    )
    
    return fig

def plot_confusion_matrix(cm):
    """Karmaşıklık matrisi görselleştirmesi"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar=True
    )
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    ax.set_title('Karmaşıklık Matrisi')
    ax.set_xticklabels(['Normal', 'Arıza'])
    ax.set_yticklabels(['Normal', 'Arıza'])
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """ROC eğrisi görselleştirmesi"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Yanlış Pozitif Oranı')
    ax.set_ylabel('Doğru Pozitif Oranı')
    ax.set_title('ROC Eğrisi')
    ax.legend(loc="lower right")
    
    return fig

def plot_sensor_data_by_failure(df, sensor_name):
    """Sensör verilerinin arıza durumuna göre dağılımı"""
    fig = px.histogram(
        df, 
        x=sensor_name, 
        color='Target',
        barmode='overlay',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        labels={'Target': 'Arıza Durumu'},
        title=f'{sensor_name} Dağılımı'
    )
    
    fig.update_layout(bargap=0.1)
    
    return fig

def main():
    """Ana Streamlit uygulaması"""
    # Başlık ve giriş
    st.markdown('<h1 class="main-header">🔧 Prediktif Bakım Analiz Panosu</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    Bu analiz panosu, endüstriyel makine bakımı için makine öğrenmesi modellerinin sonuçlarını görselleştirmektedir.
    Sensör verileri ve ürün özellikleri kullanılarak arıza tahminleri yapılmakta ve performans analizleri sunulmaktadır.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Veri yükleme
    df = load_data()
    
    if df is not None:
        # Sidebar - Filtreleme seçenekleri
        st.sidebar.markdown("## 🔍 Veri Filtreleme")
        
        # Ürün tipi filtreleme
        product_types = sorted(df['Type'].unique())
        selected_types = st.sidebar.multiselect(
            "Ürün Tipi Seçin:",
            options=product_types,
            default=product_types
        )
        
        # Alet aşınması filtreleme
        min_tool_wear = int(df['Tool wear [min]'].min())
        max_tool_wear = int(df['Tool wear [min]'].max())
        tool_wear_range = st.sidebar.slider(
            "Alet Aşınması Aralığı (dk):",
            min_value=min_tool_wear,
            max_value=max_tool_wear,
            value=(min_tool_wear, max_tool_wear)
        )
        
        # Sıcaklık aralığı filtreleme
        min_temp = float(df['Air temperature [K]'].min())
        max_temp = float(df['Air temperature [K]'].max())
        temp_range = st.sidebar.slider(
            "Hava Sıcaklığı Aralığı (K):",
            min_value=min_temp,
            max_value=max_temp,
            value=(min_temp, max_temp)
        )
        
        # Filtreleme uygula
        filtered_df = df[
            (df['Type'].isin(selected_types)) &
            (df['Tool wear [min]'] >= tool_wear_range[0]) &
            (df['Tool wear [min]'] <= tool_wear_range[1]) &
            (df['Air temperature [K]'] >= temp_range[0]) &
            (df['Air temperature [K]'] <= temp_range[1])
        ]
        
        # Sidebar - Model seçenekleri
        st.sidebar.markdown("## 🧠 Model Ayarları")
        model_name = st.sidebar.selectbox(
            "Model Seçin:",
            ["Random Forest", "Gradient Boosting"]
        )
        
        # Sidebar - Tahmin simülasyonu
        st.sidebar.markdown("## 💭 Arıza Tahmin Simülasyonu")
        
        # Kullanıcı girişleri için değerler
        air_temp = st.sidebar.number_input("Hava Sıcaklığı (K)", min_value=float(df['Air temperature [K]'].min()), 
                                           max_value=float(df['Air temperature [K]'].max()), 
                                           value=float(df['Air temperature [K]'].mean()))
        
        process_temp = st.sidebar.number_input("Süreç Sıcaklığı (K)", min_value=float(df['Process temperature [K]'].min()), 
                                               max_value=float(df['Process temperature [K]'].max()), 
                                               value=float(df['Process temperature [K]'].mean()))
        
        rotation = st.sidebar.number_input("Dönüş Hızı (rpm)", min_value=float(df['Rotational speed [rpm]'].min()), 
                                           max_value=float(df['Rotational speed [rpm]'].max()), 
                                           value=float(df['Rotational speed [rpm]'].mean()))
        
        torque = st.sidebar.number_input("Tork (Nm)", min_value=float(df['Torque [Nm]'].min()), 
                                         max_value=float(df['Torque [Nm]'].max()), 
                                         value=float(df['Torque [Nm]'].mean()))
        
        tool_wear = st.sidebar.number_input("Alet Aşınması (dk)", min_value=float(df['Tool wear [min]'].min()), 
                                            max_value=float(df['Tool wear [min]'].max()), 
                                            value=float(df['Tool wear [min]'].mean()))
        
        product_type = st.sidebar.selectbox("Ürün Tipi", options=df['Type'].unique())
        
        # Ana sekmeler
        tabs = st.tabs(["📊 Veri Analizi", "📈 Model Sonuçları", "🔮 Tahmin Simülasyonu"])
        
        with tabs[0]:
            st.markdown('<h2 class="sub-header">Veri Analizi</h2>', unsafe_allow_html=True)
            
            # Genel veri özeti
            st.markdown('<h3 class="tab-subheader">Veri Seti Özeti</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Kayıt Sayısı", len(filtered_df))
            with col2:
                st.metric("Arıza Sayısı", filtered_df['Target'].sum())
            with col3:
                ariza_orani = filtered_df['Target'].mean() * 100
                st.metric("Arıza Oranı (%)", f"{ariza_orani:.2f}%")
            
            # Veri örnekleri
            st.markdown('<h3 class="tab-subheader">Veri Örnekleri</h3>', unsafe_allow_html=True)
            st.dataframe(filtered_df.head(10), use_container_width=True)
            
            # İstatistiksel özet
            st.markdown('<h3 class="tab-subheader">İstatistiksel Özet</h3>', unsafe_allow_html=True)
            st.dataframe(filtered_df.describe().T, use_container_width=True)
            
            # Görselleştirmeler
            st.markdown('<h3 class="tab-subheader">Görselleştirmeler</h3>', unsafe_allow_html=True)
            
            # 1. Satır - Arıza dağılımı ve türleri
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_failure_distribution(filtered_df), use_container_width=True)
            with col2:
                if filtered_df['Target'].sum() > 0:
                    st.plotly_chart(plot_failure_types(filtered_df), use_container_width=True)
                else:
                    st.info("Seçilen filtrelerde arıza verisi bulunmamaktadır.")
            
            # 2. Satır - Ürün tipi ve korelasyon
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_failure_by_type(filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)
            
            # 3. Satır - 3D görselleştirme
            st.plotly_chart(plot_3d_sensor_space(filtered_df), use_container_width=True)
            
            # 4. Satır - Sensör dağılımları
            sensor_tabs = st.tabs([
                "Hava Sıcaklığı", 
                "Süreç Sıcaklığı", 
                "Dönüş Hızı", 
                "Tork", 
                "Alet Aşınması"
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
            st.markdown('<h2 class="sub-header">Model Sonuçları</h2>', unsafe_allow_html=True)
            
            # Model eğitim ve değerlendirme
            if st.button("Modeli Eğit ve Değerlendir"):
                with st.spinner("Model eğitiliyor..."):
                    # Veriyi hazırla
                    X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data(df)
                    
                    # Modeli eğit
                    pipeline = train_model(X_train, y_train, preprocessor, model_name)
                    
                    # Modeli değerlendir
                    metrics, cm, fpr, tpr, roc_auc = evaluate_model(pipeline, X_test, y_test)
                    
                    # Performans metrikleri
                    st.markdown('<h3 class="tab-subheader">Model Performans Metrikleri</h3>', unsafe_allow_html=True)
                    metric_cols = st.columns(4)
                    for i, (metric, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            st.metric(metric, f"{value:.4f}")
                    
                    # Karmaşıklık matrisi ve ROC eğrisi
                    st.markdown('<h3 class="tab-subheader">Model Değerlendirme Grafikleri</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plot_confusion_matrix(cm))
                    with col2:
                        st.pyplot(plot_roc_curve(fpr, tpr, roc_auc))
                    
                    # Özellik önemleri
                    st.markdown('<h3 class="tab-subheader">Özellik Önemi</h3>', unsafe_allow_html=True)
                    if hasattr(pipeline['classifier'], 'feature_importances_'):
                        # Özellik isimlerini al
                        feature_names = numerical_features + ['Type_M', 'Type_H']
                        importances = pipeline['classifier'].feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        # Bar plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax)
                        ax.set_title('Özellik Önem Dereceleri')
                        ax.set_xlabel('Önem Derecesi')
                        st.pyplot(fig)
                    else:
                        st.info("Seçilen model için özellik önemi hesaplanamıyor.")
            else:
                st.info("Modeli eğitmek ve değerlendirmek için 'Modeli Eğit ve Değerlendir' butonuna tıklayın.")
        
        with tabs[2]:
            st.markdown('<h2 class="sub-header">Tahmin Simülasyonu</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            Bu bölümde, sol taraftaki giriş değerlerini kullanarak yeni bir makinenin arıza olasılığını tahmin edebilirsiniz.
            Sensör değerlerini ve ürün tipini ayarlayın, ardından tahmin butonuna tıklayın.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.sidebar.button("Arıza Tahmini Yap"):
                with st.spinner("Tahmin yapılıyor..."):
                    # Veriyi hazırla
                    X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data(df)
                    
                    # Modeli eğit
                    pipeline = train_model(X_train, y_train, preprocessor, model_name)
                    
                    # Tahmin için yeni veri
                    new_data = pd.DataFrame({
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotation],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear],
                        'Type': [product_type]
                    })
                    
                    # Tahmin yap
                    prediction_proba = pipeline.predict_proba(new_data)[0][1]
                    prediction = 1 if prediction_proba > 0.5 else 0
                    
                    # Sonuçları göster
                    st.markdown('<h3 class="tab-subheader">Tahmin Sonucu</h3>', unsafe_allow_html=True)
                    
                    # Tahmin göstergeleri
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ Arıza Riski Tespit Edildi!")
                        else:
                            st.success("✅ Normal Çalışma Bekleniyor")
                    
                    with col2:
                        # İlerleme çubuğu ile arıza olasılığı
                        st.metric("Arıza Olasılığı", f"{prediction_proba:.2%}")
                        st.progress(float(prediction_proba))
                    
                    # Sensör değerleri özeti
                    st.markdown('<h3 class="tab-subheader">Girilen Parametreler</h3>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hava Sıcaklığı", f"{air_temp:.2f} K")
                        st.metric("Süreç Sıcaklığı", f"{process_temp:.2f} K")
                    with col2:
                        st.metric("Dönüş Hızı", f"{rotation:.2f} rpm")
                        st.metric("Tork", f"{torque:.2f} Nm")
                    with col3:
                        st.metric("Alet Aşınması", f"{tool_wear:.2f} dk")
                        st.metric("Ürün Tipi", product_type)
                    
                    # Arıza olasılığı göstergesi (gauge)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Arıza Olasılığı (%)", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tavsiyeler
                    st.markdown('<h3 class="tab-subheader">Bakım Tavsiyeleri</h3>', unsafe_allow_html=True)
                    
                    # Arıza durumuna göre öneriler
                    if prediction_proba > 0.75:
                        st.error("""
                        🚨 **ACİL BAKIM GEREKLİ!**
                        - Makineyi en kısa sürede durdurarak detaylı inceleme yapın
                        - Alet aşınmasını kontrol edin ve gerekirse değiştirin
                        - Sıcaklık değerlerini izleyin ve soğutma sistemini kontrol edin
                        """)
                    elif prediction_proba > 0.5:
                        st.warning("""
                        ⚠️ **PLANSIZ DURUŞ RİSKİ!**
                        - Yakın zamanda (1 hafta içinde) bakım planlanmalı
                        - Alet durumunu yakından izleyin
                        - Dönüş hızı ve tork değerlerini düzenli kontrol edin
                        """)
                    elif prediction_proba > 0.25:
                        st.info("""
                        ℹ️ **DÜŞÜK RİSK**
                        - Normal bakım programınıza devam edin
                        - Parametreleri düzenli izleyin
                        - Bir sonraki planlı bakımda alet değişimi değerlendirilebilir
                        """)
                    else:
                        st.success("""
                        ✅ **NORMAL ÇALIŞMA**
                        - Normal çalışma koşulları sürdürülebilir
                        - Rutin bakım programını takip edin
                        - Sensor verilerini düzenli kaydedin
                        """)
            else:
                st.info("Tahmin yapmak için soldaki panelden değerleri ayarlayın ve 'Arıza Tahmini Yap' butonuna tıklayın.")
        
        # Footer
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.markdown("Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi © 2024")
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("Veri yüklenemedi! Lütfen dosya yolunu kontrol edin.")

if __name__ == "__main__":
    main()