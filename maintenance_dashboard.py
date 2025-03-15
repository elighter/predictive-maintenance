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
import kagglehub
from kagglehub import KaggleDatasetAdapter

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
    .main-header {
        font-size: 2.5rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2980b9;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 0.5rem solid #3498db;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f1f8fe;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #cfe5fd;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
    }
    .tab-subheader {
        font-size: 1.2rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Kaggle'dan bakım veri setini yükler"""
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "shivamb/machine-predictive-maintenance-classification",
            "",
        )
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

@st.cache_data
def prepare_data(df):
    """Veriyi işler ve önişleme yapar"""
    # Hedef değişkeni ve özellikleri ayır
    X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']
    
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
    failure_counts = df['Machine failure'].value_counts().reset_index()
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
    failure_types = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().reset_index()
    failure_types.columns = ['Arıza Türü', 'Sayı']
    
    # Arıza türlerinin tam adları
    failure_names = {
        'TWF': 'Takım Aşınma Arızası',
        'HDF': 'Isı Dağılım Arızası',
        'PWF': 'Güç Arızası',
        'OSF': 'Aşırı Zorlanma Arızası',
        'RNF': 'Rastgele Arıza'
    }
    
    failure_types['Arıza Adı'] = failure_types['Arıza Türü'].map(failure_names)
    
    fig = px.bar(
        failure_types, 
        x='Arıza Türü', 
        y='Sayı',
        text='Sayı',
        hover_data=['Arıza Adı'],
        color='Arıza Türü',
        title='Arıza Türleri Dağılımı'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    return fig

def plot_failure_by_type(df):
    """Ürün tiplerine göre arıza oranları"""
    type_failure = df.groupby('Type')['Machine failure'].mean().reset_index()
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

def plot_feature_correlation(df):
    """Özellikler arası korelasyon matrisi"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                     'Machine failure']
    
    corr = df[numerical_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Değişkenler Arası Korelasyon Matrisi',
        aspect="auto"
    )
    
    return fig

def plot_confusion_matrix(cm, model_name):
    """Karmaşıklık matrisi görselleştirme"""
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        title=f'{model_name} Karmaşıklık Matrisi',
        labels=dict(x="Tahmin Edilen Sınıf", y="Gerçek Sınıf"),
        x=['Normal', 'Arıza'],
        y=['Normal', 'Arıza'],
    )
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """ROC eğrisi görselleştirme"""
    fig = px.line(
        x=fpr, y=tpr,
        title=f'{model_name} ROC Eğrisi (AUC = {roc_auc:.4f})',
        labels=dict(x='Yanlış Pozitif Oranı', y='Doğru Pozitif Oranı'),
    )
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_traces(line=dict(color='darkblue', width=3))
    
    return fig

def plot_feature_importance(pipeline, feature_names, model_name):
    """Özellik önem derecelerini görselleştir"""
    # Özellik önemlerini al
    if hasattr(pipeline['classifier'], 'feature_importances_'):
        importances = pipeline['classifier'].feature_importances_
        
        # Önem derecelerini değişken adlarıyla eşleştir
        feature_importance_df = pd.DataFrame({
            'Özellik': feature_names,
            'Önem Derecesi': importances
        }).sort_values('Önem Derecesi', ascending=False)
        
        fig = px.bar(
            feature_importance_df,
            x='Önem Derecesi',
            y='Özellik',
            orientation='h',
            title=f'{model_name} Özellik Önem Dereceleri',
            color='Önem Derecesi',
            color_continuous_scale='Viridis'
        )
        
        return fig
    else:
        return None

def plot_scatter_analysis(df):
    """Scatter plot analizi"""
    fig = px.scatter(
        df,
        x='Air temperature [K]',
        y='Torque [Nm]',
        color='Machine failure',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        size='Tool wear [min]',
        size_max=15,
        hover_name='UDI',
        hover_data=['Type', 'Rotational speed [rpm]', 'Process temperature [K]'],
        title='Hava Sıcaklığı, Tork ve Arıza İlişkisi'
    )
    
    return fig

def main():
    """Ana fonksiyon"""
    # Başlık
    st.markdown('<h1 class="main-header">🔧 Prediktif Bakım Analiz Panosu</h1>', unsafe_allow_html=True)
    
    # Bilgi kutusu
    st.markdown(
        '<div class="info-box">'
        'Bu panel, makine öğrenmesi kullanarak endüstriyel makinelerin bakım ihtiyaçlarını '
        'tahmin etmeye yönelik bir analiz aracıdır. '
        'Sensör verilerini analiz ederek, arıza riskini önceden tespit etmek ve '
        'planlı bakım stratejileri geliştirmek amaçlanmaktadır.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Veriyi yükle
    with st.spinner('Veriler yükleniyor...'):
        df = load_data()
    
    if df is not None:
        # Veriyi hazırla
        with st.spinner('Veriler işleniyor...'):
            X, y, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data(df)
        
        # Yan panel
        with st.sidebar:
            st.markdown('<h2 class="sub-header">Kontrol Paneli</h2>', unsafe_allow_html=True)
            
            # Model seçimi
            model_name = st.selectbox(
                "Model Seçin",
                ["Random Forest", "Gradient Boosting"]
            )
            
            # Eşik değeri
            threshold = st.slider(
                "Tahmin Eşik Değeri",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )
            
            # Özellik filtresi
            selected_features = st.multiselect(
                "Analiz İçin Özellikler",
                options=X.columns.tolist(),
                default=X.columns.tolist()[:3]
            )
            
            # Veri filtreleme
            st.markdown('<h3 class="tab-subheader">Veri Filtreleme</h3>', unsafe_allow_html=True)
            
            selected_type = st.multiselect(
                "Ürün Tipi",
                options=df["Type"].unique(),
                default=df["Type"].unique()
            )
            
            min_tool_wear, max_tool_wear = st.slider(
                "Alet Aşınması (min)",
                min_value=int(df["Tool wear [min]"].min()),
                max_value=int(df["Tool wear [min]"].max()),
                value=(int(df["Tool wear [min]"].min()), int(df["Tool wear [min]"].max()))
            )
            
            # Filtrelenmiş veri
            filtered_df = df[
                (df["Type"].isin(selected_type)) &
                (df["Tool wear [min]"] >= min_tool_wear) &
                (df["Tool wear [min]"] <= max_tool_wear)
            ]
            
            st.markdown(f"**Filtrelenmiş Veri Boyutu:** {filtered_df.shape[0]} kayıt")
            
            # Model eğitim butonu
            if st.button("Modeli Eğit ve Değerlendir"):
                with st.spinner('Model eğitiliyor...'):
                    train_flag = True
            else:
                train_flag = False
        
        # Ana panelde sekmeleri oluştur
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Veri Analizi", "🔍 Model Sonuçları", "🧪 Tahmin Simülasyonu", "ℹ️ Hakkında"])
        
        with tab1:  # Veri Analizi Sekmesi
            st.markdown('<h2 class="sub-header">Veri Analizi</h2>', unsafe_allow_html=True)
            
            # Temel metrikler
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Toplam Kayıt", len(filtered_df))
            with metric_col2:
                st.metric("Arıza Oranı", f"%{filtered_df['Machine failure'].mean()*100:.2f}")
            with metric_col3:
                failure_types = filtered_df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()
                most_common_failure = failure_types.idxmax() if failure_types.sum() > 0 else "Yok"
                st.metric("En Yaygın Arıza", most_common_failure)
            with metric_col4:
                st.metric("Ortalama Alet Aşınması", f"{filtered_df['Tool wear [min]'].mean():.1f} dk")
            
            # Grafikler
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_failure_distribution(filtered_df), use_container_width=True)
                st.plotly_chart(plot_failure_by_type(filtered_df), use_container_width=True)
            
            with col2:
                st.plotly_chart(plot_failure_types(filtered_df), use_container_width=True)
                st.plotly_chart(plot_feature_correlation(filtered_df), use_container_width=True)
            
            # Scatter plot analizi
            st.markdown('<h3 class="tab-subheader">Sensör Verileri Analizi</h3>', unsafe_allow_html=True)
            st.plotly_chart(plot_scatter_analysis(filtered_df), use_container_width=True)
            
            # Ham veri gösterimi
            st.markdown('<h3 class="tab-subheader">Ham Veri</h3>', unsafe_allow_html=True)
            st.dataframe(filtered_df.head(50), use_container_width=True)
        
        with tab2:  # Model Sonuçları Sekmesi
            st.markdown('<h2 class="sub-header">Model Sonuçları</h2>', unsafe_allow_html=True)
            
            if train_flag:
                # Modeli eğit
                with st.spinner(f"{model_name} modeli eğitiliyor..."):
                    pipeline = train_model(X_train, y_train, preprocessor, model_name)
                
                # Modeli değerlendir
                with st.spinner("Model değerlendiriliyor..."):
                    metrics, cm, fpr, tpr, roc_auc = evaluate_model(pipeline, X_test, y_test)
                
                # Model metrikleri
                st.markdown('<h3 class="tab-subheader">Model Performans Metrikleri</h3>', unsafe_allow_html=True)
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Doğruluk", f"{metrics['Doğruluk']:.4f}")
                with metric_col2:
                    st.metric("Kesinlik", f"{metrics['Kesinlik']:.4f}")
                with metric_col3:
                    st.metric("Duyarlılık", f"{metrics['Duyarlılık']:.4f}")
                with metric_col4:
                    st.metric("F1 Skoru", f"{metrics['F1 Skoru']:.4f}")
                
                # Karmaşıklık matrisi ve ROC
                st.markdown('<h3 class="tab-subheader">Model Değerlendirme Grafikleri</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_confusion_matrix(cm, model_name), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_roc_curve(fpr, tpr, roc_auc, model_name), use_container_width=True)
                
                # Özellik önemleri
                st.markdown('<h3 class="tab-subheader">Özellik Önemleri</h3>', unsafe_allow_html=True)
                
                # İşlenmiş özelliklerin adlarını al
                feature_names = X.columns.tolist()
                # OneHotEncoder ile dönüştürülmüş kategorik değişkenler için özellik isimlerini güncelle
                if len(categorical_features) > 0:
                    for cat_feature in categorical_features:
                        feature_names.remove(cat_feature)  # Orijinal kategorik değişkeni kaldır
                        # Type_L ve Type_M eklenir (Type_H drop=first nedeniyle dahil edilmez)
                        if cat_feature == 'Type':
                            feature_names.extend(['Type_L', 'Type_M'])
                
                feature_imp_fig = plot_feature_importance(pipeline, feature_names, model_name)
                if feature_imp_fig:
                    st.plotly_chart(feature_imp_fig, use_container_width=True)
                else:
                    st.warning("Seçilen model özellik önem derecelerini desteklemiyor.")
            else:
                st.info("Model henüz eğitilmedi. Modeli eğitmek için sol paneldeki 'Modeli Eğit ve Değerlendir' butonuna tıklayın.")
        
        with tab3:  # Tahmin Simülasyonu Sekmesi
            st.markdown('<h2 class="sub-header">Tahmin Simülasyonu</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">'
                      'Bu bölümde, kendi belirlediğiniz sensör değerleriyle makine arızası tahmininde bulunabilirsiniz. '
                      'Değerleri interaktif olarak ayarlayın ve modelin tahminini görün.'
                      '</div>',
                      unsafe_allow_html=True)
            
            if train_flag:
                # Kullanıcıdan değerler al
                st.markdown('<h3 class="tab-subheader">Sensör Değerlerini Ayarlayın</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    air_temp = st.slider(
                        "Hava Sıcaklığı (K)",
                        min_value=float(df["Air temperature [K]"].min()),
                        max_value=float(df["Air temperature [K]"].max()),
                        value=float(df["Air temperature [K]"].mean()),
                        step=0.1
                    )
                    
                    process_temp = st.slider(
                        "İşlem Sıcaklığı (K)",
                        min_value=float(df["Process temperature [K]"].min()),
                        max_value=float(df["Process temperature [K]"].max()),
                        value=float(df["Process temperature [K]"].mean()),
                        step=0.1
                    )
                
                with col2:
                    rot_speed = st.slider(
                        "Dönüş Hızı (rpm)",
                        min_value=int(df["Rotational speed [rpm]"].min()),
                        max_value=int(df["Rotational speed [rpm]"].max()),
                        value=int(df["Rotational speed [rpm]"].mean()),
                        step=10
                    )
                    
                    torque = st.slider(
                        "Tork (Nm)",
                        min_value=float(df["Torque [Nm]"].min()),
                        max_value=float(df["Torque [Nm]"].max()),
                        value=float(df["Torque [Nm]"].mean()),
                        step=0.1
                    )
                
                with col3:
                    tool_wear = st.slider(
                        "Alet Aşınması (min)",
                        min_value=int(df["Tool wear [min]"].min()),
                        max_value=int(df["Tool wear [min]"].max()),
                        value=int(df["Tool wear [min]"].mean()),
                        step=1
                    )
                    
                    machine_type = st.selectbox(
                        "Makine Tipi",
                        options=df["Type"].unique(),
                        index=0
                    )
                
                # Tahmin butonu
                if st.button("Arıza Tahminini Göster"):
                    # Tek satırlık veri oluştur
                    input_data = pd.DataFrame({
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rot_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear],
                        'Type': [machine_type]
                    })
                    
                    # Tahmin yap
                    pred_proba = pipeline.predict_proba(input_data)[0, 1]
                    pred_class = 1 if pred_proba >= threshold else 0
                    
                    # Sonuçları göster
                    st.markdown('<h3 class="tab-subheader">Tahmin Sonuçları</h3>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pred_class == 1:
                            st.error("⚠️ ARIZA RİSKİ TESPİT EDİLDİ!")
                            st.markdown(f"<h1 style='color: #E74C3C; text-align: center;'>%{pred_proba*100:.1f}</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Arıza Olasılığı</p>", unsafe_allow_html=True)
                        else:
                            st.success("✅ NORMAL ÇALIŞMA")
                            st.markdown(f"<h1 style='color: #2ECC71; text-align: center;'>%{(1-pred_proba)*100:.1f}</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Normal Çalışma Olasılığı</p>", unsafe_allow_html=True)
                    
                    with col2:
                        # Gauge chart ile olasılık gösterimi
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = pred_proba * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Arıza Riski (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold * 100
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Önerileri göster
                    st.markdown('<h3 class="tab-subheader">Bakım Önerileri</h3>', unsafe_allow_html=True)
                    
                    if pred_class == 1:
                        if tool_wear > 150:
                            st.warning("👉 Alet aşınması yüksek (>150 dk). Aletin değiştirilmesi önerilir.")
                        
                        if process_temp - air_temp > 15:
                            st.warning("👉 Sıcaklık farkı kritik seviyede. Soğutma sistemini kontrol edin.")
                        
                        if rot_speed > 2300:
                            st.warning("👉 Dönüş hızı yüksek. Devir sayısını düşürmeyi değerlendirin.")
                        
                        if machine_type == 'H':
                            st.warning("👉 H tipi makineler daha yüksek arıza riski taşır. Daha sık kontrol önerilir.")
                        
                        st.error("⚠️ Acil bakım planlanması önerilir!")
                    else:
                        suggestions = []
                        
                        if tool_wear > 100:
                            suggestions.append("Alet aşınması izlenmeli (şu an güvenli aralıkta).")
                        
                        if machine_type == 'H' and pred_proba > 0.3:
                            suggestions.append("H tipi makine olduğu için düzenli kontroller sürdürülmeli.")
                        
                        if not suggestions:
                            st.success("✅ Makine normal parametrelerle çalışıyor. Rutin bakım yeterli.")
                        else:
                            for suggestion in suggestions:
                                st.info(f"👉 {suggestion}")
                            
                            st.success("✅ Acil bakım gerekmiyor, ancak belirtilen noktalara dikkat edilmeli.")
            else:
                st.warning("Tahminde bulunmak için önce modeli eğitmeniz gerekmektedir. Sol paneldeki 'Modeli Eğit ve Değerlendir' butonuna tıklayın.")
        
        with tab4:  # Hakkında Sekmesi
            st.markdown('<h2 class="sub-header">Proje Hakkında</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            ### 🔧 Prediktif Bakım Nedir?
            
            Prediktif bakım, makinelerin ne zaman arızalanacağını önceden tahmin ederek, 
            bakım faaliyetlerini planlama yaklaşımıdır. Geleneksel "arızalanınca tamir et" veya 
            "belirli aralıklarla kontrol et" stratejileri yerine, veriye dayalı olarak 
            "ihtiyaç olduğunda bakım yap" prensibini uygular.
            
            ### 📊 Bu Projede Kullanılan Veri Seti
            
            Bu projede kullanılan veri seti, endüstriyel makinelerden toplanan sensör verilerini ve 
            arıza kayıtlarını içermektedir. Veri setinde şu değişkenler bulunur:
            
            - **UDI**: Benzersiz tanımlayıcı
            - **Product ID**: Ürün kimliği
            - **Type**: Ürün tipi (L: Düşük, M: Orta, H: Yüksek)
            - **Air temperature [K]**: Ortam hava sıcaklığı (Kelvin)
            - **Process temperature [K]**: İşlem sıcaklığı (Kelvin)
            - **Rotational speed [rpm]**: Dönüş hızı (dakikada devir)
            - **Torque [Nm]**: Tork değeri (Newton metre)
            - **Tool wear [min]**: Alet aşınması (dakika)
            - **Machine failure**: Makine arızası (0: Arıza yok, 1: Arıza var)
            - **TWF, HDF, PWF, OSF, RNF**: Farklı arıza türleri
            
            ### 💡 Prediktif Bakımın Faydaları
            
            - **Maliyet Tasarrufu**: Bakım maliyetlerinde %10-40 azalma
            - **Ekipman Ömrü**: Makine ömründe %20'ye varan artış
            - **Üretim Verimliliği**: Plansız duruşlarda %50'ye varan azalma
            - **İşgücü Optimizasyonu**: Bakım işçiliğinde %10-15 tasarruf
            
            ### 🔬 Kullanılan Teknolojiler
            
            - **Python**: Veri analizi ve model geliştirme
            - **Scikit-learn**: Makine öğrenmesi algoritmaları
            - **Pandas & NumPy**: Veri manipülasyonu
            - **Plotly & Matplotlib**: Veri görselleştirme
            - **Streamlit**: İnteraktif dashboard geliştirme
            """)
    else:
        st.error("Veri yüklenemedi. Lütfen bağlantınızı kontrol edin ve sayfayı yenileyin.")
    
    # Footer
    st.markdown(
        '<div class="footer">'
        'Prediktif Bakım Analiz Panosu • Makine Öğrenmesi Projesi • '
        'Geliştiren: Prediktif Bakım Ekibi'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 