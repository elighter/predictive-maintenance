#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi 
Model Yorumlanabilirlik Modülü

Bu modül, makine bakımı arıza tahmin modellerinin yorumlanabilirliğini 
SHAP değerleri ve PDP grafikleri ile artırmak için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.inspection import permutation_importance, partial_dependence
# from sklearn.inspection import plot_partial_dependence  # Eski sürümlerde vardı

# SHAP kütüphanesi
import shap

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

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

def prepare_data(df):
    """Veriyi eğitim ve test için hazırlar"""
    # UDI, Product ID ve Target değişkenlerini çıkaralım
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    y = df['Target']
    
    # Eğitim ve test setlerine ayıralım
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Kategorik ve sayısal değişkenleri ayıralım
    categorical_features = ['Type']
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Önişleme pipeline'ı oluşturalım
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # Verileri dönüştürelim
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Özellik isimlerini alalım (SHAP grafikleri için)
    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(ohe_features)
    
    # Dönüştürülmüş verileri DataFrame olarak oluşturalım (SHAP için)
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    numerical_features_idx = list(range(len(numerical_features)))
    
    return (X_train, X_train_processed, X_train_df, X_test, X_test_processed, X_test_df, 
            y_train, y_test, feature_names, numerical_features_idx)

def train_models(X_train_processed, y_train):
    """Farklı modelleri eğitir"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        print(f"{name} modeli eğitildi.")
    
    return models

def evaluate_models(models, X_test_processed, y_test):
    """Modelleri değerlendirir ve performans metriklerini döndürür"""
    results = {}
    
    for name, model in models.items():
        # Tahminler
        y_pred = model.predict(X_test_processed)
        
        # Performans metrikleri
        results[name] = {
            'Doğruluk': accuracy_score(y_test, y_pred),
            'Kesinlik': precision_score(y_test, y_pred),
            'Duyarlılık': recall_score(y_test, y_pred),
            'F1 Skoru': f1_score(y_test, y_pred)
        }
        
        print(f"\n{name} Model Performansı:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))
    
    return results

def plot_confusion_matrices(models, X_test_processed, y_test):
    """Modellerin karmaşıklık matrislerini görselleştirir"""
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_processed)
        cm = confusion_matrix(y_test, y_pred)
        
        ax = axes[i] if len(models) > 1 else axes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} Karmaşıklık Matrisi')
        ax.set_xlabel('Tahmin Edilen Sınıf')
        ax.set_ylabel('Gerçek Sınıf')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Karmaşıklık matrisleri kaydedildi: confusion_matrices.png")

def plot_feature_importance(models, X_train, feature_names):
    """Model özellik önem derecelerini görselleştirir"""
    fig, axes = plt.subplots(1, len(models), figsize=(18, 8))
    
    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, 'feature_importances_'):
            # Özellik önem değerlerini al
            importances = model.feature_importances_
            
            # Önem derecelerine göre sırala
            indices = np.argsort(importances)[::-1]
            
            ax = axes[i] if len(models) > 1 else axes
            
            # Bar plot oluştur
            sns.barplot(x=importances[indices][:10], y=[feature_names[j] for j in indices][:10], ax=ax)
            
            ax.set_title(f'{name} Özellik Önem Dereceleri')
            ax.set_xlabel('Önem Derecesi')
            ax.set_ylabel('Özellik')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Özellik önem dereceleri kaydedildi: feature_importance.png")

def plot_shap_values(models, X_train_df, X_test_df):
    """SHAP değerleri kullanarak model kararlarını görselleştirir"""
    for name, model in models.items():
        print(f"\n{name} için SHAP değerleri hesaplanıyor...")
        
        # SHAP açıklayıcısını oluştur
        explainer = shap.Explainer(model)
        
        # Eğitim verileri için SHAP değerlerini hesapla
        shap_values = explainer(X_train_df)
        
        # SHAP önem grafiği
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
        plt.title(f"{name} - SHAP Özellik Önem Grafiği", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'shap_importance_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP özet grafiği
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_df, show=False)
        plt.title(f"{name} - SHAP Özet Grafiği", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'shap_summary_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Birkaç test örneği için karar değişim grafikleri
        try:
            # İlk 3 test örneği için SHAP açıklaması
            for i in range(min(3, len(X_test_df))):
                plt.figure(figsize=(12, 6))
                # Decision plot (daha basit ve daha az hata olasılığı var)
                shap.decision_plot(explainer.expected_value, shap_values.values[:1], X_train_df.iloc[:1], show=False)
                plt.title(f"{name} - Örnek {i+1} SHAP Karar Grafiği", fontsize=14)
                plt.tight_layout()
                plt.savefig(f'shap_decision_{name.replace(" ", "_").lower()}_{i+1}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Karar grafiği oluşturulurken hata: {e}")
            print("SHAP karar grafiği atlanıyor...")

def plot_pdp(models, X_train, feature_names, numerical_features_idx):
    """
    Kısmi Bağımlılık Grafikleri (PDP) çizerek modellerin 
    her bir özelliğe nasıl tepki verdiğini görselleştirir
    """
    try:
        n_features = len(numerical_features_idx)
        
        plt.figure(figsize=(15, n_features * 5))
        
        # Her sayısal özellik için
        for i, idx in enumerate(numerical_features_idx):
            feature_name = feature_names[idx]
            
            plt.subplot(n_features, 1, i+1)
            
            # Her model için
            for model_name, model in models.items():
                try:
                    # Kısmi bağımlılık hesapla
                    pdp_result = partial_dependence(
                        model, X_train, [idx], 
                        kind="average", grid_resolution=50
                    )
                    
                    # Sonuçları çiz - yeni API formatına göre
                    if "values" in pdp_result:
                        # Eski format
                        feature_values = pdp_result["values"][0]
                        pdp_values = pdp_result["average"][0]
                    else:
                        # Yeni format
                        feature_values = pdp_result["grid_values"][0]
                        pdp_values = pdp_result["average_dependence"][0]
                    
                    plt.plot(feature_values, pdp_values, 
                            label=f'{model_name}', 
                            linewidth=2, 
                            marker='o',
                            markersize=4)
                
                except Exception as e:
                    print(f"PDP oluşturulurken hata ({model_name}, {feature_name}): {e}")
                    continue
            
            plt.title(f'{feature_name} için Kısmi Bağımlılık', fontsize=14)
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel('Arıza Olasılığı', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pdp_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Kısmi bağımlılık grafikleri kaydedildi: pdp_plots.png")
    
    except Exception as e:
        print(f"Kısmi bağımlılık grafikleri oluşturulurken hata: {e}")

def main():
    """Ana fonksiyon"""
    print("Endüstriyel Bakım Model Yorumlanabilirlik Uygulaması")
    
    # Veri setini yükle
    df = load_maintenance_data()
    
    if df is not None:
        # Veriyi hazırla
        data = prepare_data(df)
        (X_train, X_train_processed, X_train_df, X_test, X_test_processed, X_test_df, 
         y_train, y_test, feature_names, numerical_features_idx) = data
        
        # Modelleri eğit
        models = train_models(X_train_processed, y_train)
        
        # Modelleri değerlendir
        results = evaluate_models(models, X_test_processed, y_test)
        
        # Karmaşıklık matrisleri
        plot_confusion_matrices(models, X_test_processed, y_test)
        
        # Özellik önem dereceleri
        plot_feature_importance(models, X_train, feature_names)
        
        # SHAP değerleri
        plot_shap_values(models, X_train_df, X_test_df)
        
        # PDP grafikleri
        plot_pdp(models, X_train, feature_names, numerical_features_idx)
        
        print("\nTüm model yorumlama grafikleri oluşturuldu!")
        
    else:
        print("Veri yüklenemediği için analizler yapılamadı.")

if __name__ == "__main__":
    main()