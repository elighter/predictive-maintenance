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
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

def load_maintenance_data():
    """Bakım veri setini yerel dosyadan yükler"""
    try:
        file_path = "data/predictive_maintenance.csv"
        df = pd.read_csv(file_path)
        print(f"Veri seti başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata: {e}")
        return None

def plot_failure_distribution(df):
    """Arıza dağılımını gösteren pasta grafiği oluşturur"""
    # Arıza dağılımı
    failure_counts = df['Target'].value_counts().reset_index()
    failure_counts.columns = ['Durum', 'Sayı']
    failure_counts['Durum'] = failure_counts['Durum'].map({0: 'Normal', 1: 'Arıza'})
    
    # Pasta grafiği
    fig = px.pie(
        failure_counts, 
        values='Sayı', 
        names='Durum',
        title='Makine Arıza Dağılımı',
        color='Durum',
        color_discrete_map={'Normal': '#2ecc71', 'Arıza': '#e74c3c'},
        hole=0.4
    )
    fig.update_traces(textinfo='percent+value', textfont_size=14)
    fig.update_layout(
        title_font_size=20,
        showlegend=True,
        legend=dict(orientation='h', yanchor='top', y=-0.1)
    )
    
    return fig

def plot_failure_types(df):
    """Arıza türleri dağılımı görselleştirmesi"""
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

def plot_3d_sensor_space(df):
    """3D sensör uzayı görselleştirmesi"""
    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='Rotational speed [rpm]',
        y='Torque [Nm]',
        z='Tool wear [min]',
        color='Target',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        opacity=0.7,
        size='Tool wear [min]',
        size_max=10,
        symbol='Type',
        title='3D Sensör Uzayında Makine Durumu',
        labels={
            'Rotational speed [rpm]': 'Dönüş Hızı (rpm)',
            'Torque [Nm]': 'Tork (Nm)',
            'Tool wear [min]': 'Alet Aşınması (dk)',
            'Target': 'Durum',
            'Type': 'Ürün Tipi'
        }
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Dönüş Hızı (rpm)',
            yaxis_title='Tork (Nm)',
            zaxis_title='Alet Aşınması (dk)'
        ),
        title_font_size=20,
        height=800,
        legend=dict(title="Durum")
    )
    
    return fig

def plot_interactive_correlation(df):
    """İnteraktif korelasyon matrisi görselleştirmesi"""
    # Korelasyon için sadece sayısal değişkenleri seçelim
    numeric_columns = ['Air temperature [K]', 'Process temperature [K]', 
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                      'Target']
    
    # Korelasyon matrisi
    corr = df[numeric_columns].corr().round(2)
    
    # Heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Değişkenler Arası Korelasyon Matrisi"
    )
    
    fig.update_layout(
        title_font_size=20,
        height=700,
        width=800
    )
    
    return fig

def plot_failure_by_type(df):
    """Ürün tiplerine göre arıza oranları görselleştirmesi"""
    # Ürün tipine göre arıza oranı
    type_failure = df.groupby('Type')['Target'].mean().reset_index()
    type_failure['Arıza Oranı (%)'] = type_failure['Target'] * 100
    
    # Bar grafiği
    fig = px.bar(
        type_failure,
        x='Type',
        y='Arıza Oranı (%)',
        color='Type',
        text='Arıza Oranı (%)',
        title='Ürün Tiplerine Göre Arıza Oranları',
        labels={'Type': 'Ürün Tipi', 'Arıza Oranı (%)': 'Arıza Oranı (%)'},
        color_discrete_map={'L': '#3498db', 'M': '#f1c40f', 'H': '#e74c3c'}
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        textfont=dict(size=14)
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=600,
        width=800,
        xaxis=dict(categoryorder='total descending')
    )
    
    # Y eksenini yüzde formatında gösterelim
    fig.update_yaxes(ticksuffix="%")
    
    return fig

def plot_sensor_distributions(df):
    """Sensör değerlerinin arıza durumuna göre dağılımı görselleştirmesi"""
    sensors = ['Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Alt grafikleri oluştur
    fig = make_subplots(rows=len(sensors), cols=1, 
                        subplot_titles=[f"{sensor} Dağılımı" for sensor in sensors],
                        vertical_spacing=0.05)
    
    # Her sensör için histogram
    for i, sensor in enumerate(sensors):
        # Normal durum histogramı
        fig.add_trace(
            go.Histogram(
                x=df[df['Target'] == 0][sensor],
                name="Normal",
                marker_color='#2ecc71',
                opacity=0.7,
                histnorm='probability',
                nbinsx=30
            ),
            row=i+1, col=1
        )
        
        # Arıza durumu histogramı
        fig.add_trace(
            go.Histogram(
                x=df[df['Target'] == 1][sensor],
                name="Arıza",
                marker_color='#e74c3c',
                opacity=0.7,
                histnorm='probability',
                nbinsx=30
            ),
            row=i+1, col=1
        )
        
        # Her grafik için başlık ekleyelim
        fig.update_xaxes(title_text=sensor, row=i+1, col=1)
        fig.update_yaxes(title_text="Olasılık", row=i+1, col=1)
    
    # Grafik düzeni
    fig.update_layout(
        title_text="Sensör Değerlerinin Arıza Durumuna Göre Dağılımı",
        title_font_size=20,
        height=1000,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_animated_temp_torque(df):
    """Sıcaklık ve tork ilişkisinin alet aşınmasına göre animasyonlu görselleştirmesi"""
    # Alet aşınması için aralıklar belirleyelim
    tool_wear_ranges = np.linspace(min(df['Tool wear [min]']), max(df['Tool wear [min]']), 20)
    
    # Her aralık için frame oluşturarak animasyon hazırlayalım
    frames = []
    for i in range(len(tool_wear_ranges)-1):
        low, high = tool_wear_ranges[i], tool_wear_ranges[i+1]
        
        # Aralık içindeki verileri filtreleyelim
        mask = (df['Tool wear [min]'] >= low) & (df['Tool wear [min]'] < high)
        frame_data = df[mask]
        
        # Frame
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=frame_data[frame_data['Target'] == 0]['Air temperature [K]'],
                    y=frame_data[frame_data['Target'] == 0]['Torque [Nm]'],
                    mode='markers',
                    marker=dict(color='#2ecc71', size=10, opacity=0.7),
                    name='Normal'
                ),
                go.Scatter(
                    x=frame_data[frame_data['Target'] == 1]['Air temperature [K]'],
                    y=frame_data[frame_data['Target'] == 1]['Torque [Nm]'],
                    mode='markers',
                    marker=dict(color='#e74c3c', size=10, opacity=0.7),
                    name='Arıza'
                )
            ],
            name=f'Alet Aşınması: {low:.1f}-{high:.1f} dk'
        )
        frames.append(frame)
    
    # İlk veri noktalarını gösterecek scatter plot
    scatter_normal = go.Scatter(
        x=df[df['Target'] == 0]['Air temperature [K]'].iloc[:10],
        y=df[df['Target'] == 0]['Torque [Nm]'].iloc[:10],
        mode='markers',
        marker=dict(color='#2ecc71', size=10, opacity=0.7),
        name='Normal'
    )
    
    scatter_failure = go.Scatter(
        x=df[df['Target'] == 1]['Air temperature [K]'].iloc[:10],
        y=df[df['Target'] == 1]['Torque [Nm]'].iloc[:10],
        mode='markers',
        marker=dict(color='#e74c3c', size=10, opacity=0.7),
        name='Arıza'
    )
    
    # Ana figür
    fig = go.Figure(
        data=[scatter_normal, scatter_failure],
        frames=frames
    )
    
    # Animasyon kontrolleri ekle
    fig.update_layout(
        title='Sıcaklık ve Tork İlişkisinin Alet Aşınmasına Göre Değişimi',
        title_font_size=20,
        xaxis_title='Hava Sıcaklığı (K)',
        yaxis_title='Tork (Nm)',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        height=700,
        width=900,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Oynat",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, 
                                     "fromcurrent": True}]
                    ),
                    dict(
                        label="Durdur",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": True}, 
                                       "mode": "immediate", "transition": {"duration": 0}}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f.name],
                            {"frame": {"duration": 300, "redraw": True}, 
                             "mode": "immediate", "transition": {"duration": 300}}
                        ],
                        label=f.name
                    )
                    for f in frames
                ],
                active=0,
                transition={"duration": 300},
                x=0.1,
                y=0,
                currentvalue={
                    "font": {"size": 12},
                    "prefix": "Aşınma: ",
                    "visible": True,
                    "xanchor": "right"
                },
                len=0.9
            )
        ]
    )
    
    return fig

def main():
    """Ana fonksiyon"""
    print("Endüstriyel Bakım İnteraktif Görselleştirme Uygulaması")
    
    # Veri setini yükle
    df = load_maintenance_data()
    
    if df is not None:
        # Arıza dağılımı
        fig_failure_dist = plot_failure_distribution(df)
        fig_failure_dist.write_html("ariza_dagilimi.html")
        print("ariza_dagilimi.html dosyası oluşturuldu.")
        
        # Arıza türleri dağılımı
        fig_failure_types = plot_failure_types(df)
        fig_failure_types.write_html("ariza_turleri.html")
        print("ariza_turleri.html dosyası oluşturuldu.")
        
        # 3D sensör uzayı
        fig_3d = plot_3d_sensor_space(df)
        fig_3d.write_html("3d_sensor_uzayi.html")
        print("3d_sensor_uzayi.html dosyası oluşturuldu.")
        
        # Korelasyon matrisi
        fig_corr = plot_interactive_correlation(df)
        fig_corr.write_html("korelasyon_matrisi.html")
        print("korelasyon_matrisi.html dosyası oluşturuldu.")
        
        # Ürün tiplerine göre arıza oranları
        fig_type = plot_failure_by_type(df)
        fig_type.write_html("urun_tipleri_ariza.html")
        print("urun_tipleri_ariza.html dosyası oluşturuldu.")
        
        # Sensör dağılımları
        fig_sensors = plot_sensor_distributions(df)
        fig_sensors.write_html("sensor_dagilimi.html")
        print("sensor_dagilimi.html dosyası oluşturuldu.")
        
        # Animasyonlu sıcaklık-tork ilişkisi
        fig_animated = plot_animated_temp_torque(df)
        fig_animated.write_html("sicaklik_tork_animasyon.html")
        print("sicaklik_tork_animasyon.html dosyası oluşturuldu.")
        
        print("Tüm görselleştirmeler başarıyla oluşturuldu!")
        
    else:
        print("Veri yüklenemediği için görselleştirmeler oluşturulamadı.")

if __name__ == "__main__":
    main()