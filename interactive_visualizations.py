#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi 
İnteraktif Görselleştirme Modülü

Bu modül, makine bakımı veri seti için Plotly kullanarak 
interaktif görselleştirmeler sunmaktadır.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_maintenance_data():
    """Kaggle'dan bakım veri setini yükler"""
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "shivamb/machine-predictive-maintenance-classification",
            "",
        )
        print(f"Veri seti başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata: {e}")
        return None

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
    
    fig.update_layout(
        title_font_size=22,
        legend_title_font_size=16,
        font=dict(size=14),
        height=600,
        width=800
    )
    
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
        color='Sayı',
        color_continuous_scale='Viridis',
        title='Arıza Türleri Dağılımı'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    fig.update_layout(
        title_font_size=22,
        xaxis_title='Arıza Türü',
        yaxis_title='Arıza Sayısı',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=600,
        width=900
    )
    
    return fig

def plot_3d_sensor_space(df):
    """3D sensör uzayında makinelerin dağılımı"""
    fig = px.scatter_3d(
        df, 
        x='Air temperature [K]', 
        y='Rotational speed [rpm]', 
        z='Torque [Nm]',
        color='Machine failure',
        symbol='Type',
        size='Tool wear [min]',
        size_max=15,
        opacity=0.7,
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        title='Makinelerin 3D Sensör Uzayındaki Dağılımı'
    )
    
    fig.update_layout(
        title_font_size=20,
        scene=dict(
            xaxis_title='Hava Sıcaklığı (K)',
            yaxis_title='Dönüş Hızı (rpm)',
            zaxis_title='Tork (Nm)',
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            zaxis_title_font_size=14,
        ),
        height=800,
        width=1000
    )
    
    return fig

def plot_interactive_correlation(df):
    """İnteraktif korelasyon matrisi"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                     'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    corr = df[numerical_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Değişkenler Arası Korelasyon Matrisi',
        aspect="auto"
    )
    
    fig.update_layout(
        title_font_size=20,
        height=800,
        width=900
    )
    
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
        title='Ürün Tiplerine Göre Arıza Oranları (%)',
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    
    fig.update_layout(
        title_font_size=22,
        xaxis_title='Ürün Tipi',
        yaxis_title='Arıza Oranı (%)',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=600,
        width=800
    )
    
    return fig

def plot_sensor_distributions(df):
    """Sensör değerlerinin arıza durumuna göre dağılımı"""
    sensors = ['Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # 2x3 subplot oluştur
    fig = make_subplots(rows=2, cols=3, subplot_titles=sensors)
    
    colors = {'Normal': '#2ECC71', 'Arıza': '#E74C3C'}
    
    for i, sensor in enumerate(sensors):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Normal makineler için histogram
        fig.add_trace(
            go.Histogram(
                x=df[df['Machine failure'] == 0][sensor],
                name='Normal',
                opacity=0.7,
                marker_color=colors['Normal'],
                nbinsx=30
            ),
            row=row, col=col
        )
        
        # Arızalı makineler için histogram
        fig.add_trace(
            go.Histogram(
                x=df[df['Machine failure'] == 1][sensor],
                name='Arıza',
                opacity=0.7,
                marker_color=colors['Arıza'],
                nbinsx=30
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Sensör Değerlerinin Arıza Durumuna Göre Dağılımı',
        title_font_size=20,
        barmode='overlay',
        height=800,
        width=1200,
        showlegend=True
    )
    
    return fig

def plot_animated_temperature_torque(df):
    """Hava sıcaklığı ve tork ilişkisinin animasyonu"""
    df_copy = df.copy()
    
    # Tool wear aralıklarına göre gruplama
    bins = [0, 50, 100, 150, 200, 250]
    labels = ['0-50', '51-100', '101-150', '151-200', '201-250']
    df_copy['Tool wear range'] = pd.cut(df_copy['Tool wear [min]'], bins=bins, labels=labels)
    
    fig = px.scatter(
        df_copy,
        x='Air temperature [K]',
        y='Torque [Nm]',
        color='Machine failure',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        size='Process temperature [K]',
        size_max=15,
        hover_name='UDI',
        hover_data=['Type', 'Rotational speed [rpm]', 'Tool wear [min]'],
        animation_frame='Tool wear range',
        title='Hava Sıcaklığı ve Tork İlişkisi (Alet Aşınmasına Göre Animasyon)'
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title='Hava Sıcaklığı (K)',
        yaxis_title='Tork (Nm)',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=700,
        width=1000
    )
    
    return fig

def main():
    """Ana fonksiyon"""
    df = load_maintenance_data()
    
    if df is not None:
        # Tüm grafikleri oluştur
        print("İnteraktif grafikler oluşturuluyor...")
        
        # Her grafiği oluştur ve HTML olarak kaydet
        plots = {
            "ariza_dagilimi": plot_failure_distribution(df),
            "ariza_turleri": plot_failure_types(df),
            "3d_sensor_uzayi": plot_3d_sensor_space(df),
            "korelasyon_matrisi": plot_interactive_correlation(df),
            "urun_tipleri_ariza": plot_failure_by_type(df),
            "sensor_dagilimi": plot_sensor_distributions(df),
            "sicaklik_tork_animasyon": plot_animated_temperature_torque(df)
        }
        
        print("Toplam 7 interaktif grafik oluşturuldu!")
        
        # Grafikleri HTML dosyalarına kaydet
        for name, plot in plots.items():
            plot.write_html(f"{name}.html")
            print(f"{name}.html dosyası kaydedildi.")
        
        print("Tüm grafikler başarıyla kaydedildi!")
    else:
        print("Veri yüklenemediği için grafikler oluşturulamadı.")

if __name__ == "__main__":
    main() 