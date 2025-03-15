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
from sklearn.inspection import permutation_importance, partial_dependence, plot_partial_dependence

# SHAP kütüphanesi
import shap

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

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

def prepare_data(df):
    """Veriyi makine öğrenmesi için hazırlar"""
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
    
    # Pipeline'ı fit et
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # İşlenmiş verileri DataFrame'e dönüştür (SHAP görselleştirmeleri için)
    one_hot_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = one_hot_encoder.get_feature_names_out(['Type'])
    
    processed_feature_names = list(numerical_features) + list(cat_feature_names)
    
    X_train_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    
    return X_train, X_test, X_train_processed, X_test_processed, X_train_df, X_test_df, y_train, y_test, preprocessor

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
    """SHAP değerlerini görselleştirir"""
    for name, model in models.items():
        print(f"\n{name} için SHAP değerleri hesaplanıyor...")
        
        # SHAP açıklayıcı oluştur
        if name == 'Random Forest':
            explainer = shap.TreeExplainer(model)
        elif name == 'Gradient Boosting':
            explainer = shap.TreeExplainer(model)
        else:
            print(f"{name} için SHAP desteği bulunmuyor, atlanıyor.")
            continue
        
        # SHAP değerleri hesapla (örnek olarak test verisinin ilk 100 satırı)
        sample_size = min(100, X_test_df.shape[0])
        shap_values = explainer.shap_values(X_test_df.iloc[:sample_size])
        
        # Özet grafiği (tüm özelliklerin toplam etkileri)
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_df.iloc[:sample_size], plot_type="bar", 
                         show=False, color='#2574A9')
        plt.title(f'{name} SHAP Özellik Önem Dereceleri')
        plt.tight_layout()
        plt.savefig(f'shap_importance_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Özet noktasal grafik (her örnek için SHAP değerleri)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_df.iloc[:sample_size], show=False)
        plt.title(f'{name} SHAP Değerleri Dağılımı')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # İlk 5 örnek için kuvvet grafiği
        for i in range(min(5, sample_size)):
            plt.figure(figsize=(15, 5))
            shap.force_plot(explainer.expected_value, shap_values[i], X_test_df.iloc[i], 
                          matplotlib=True, show=False)
            plt.title(f'{name} Model SHAP Kuvvet Grafiği - Örnek {i+1}')
            plt.tight_layout()
            plt.savefig(f'shap_force_plot_{name.replace(" ", "_").lower()}_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"{name} için SHAP grafikleri oluşturuldu.")

def plot_pdp(models, X_train, feature_names, numerical_features_idx):
    """Partial Dependence Plot (PDP) grafiklerini oluşturur"""
    for name, model in models.items():
        print(f"\n{name} için PDP grafikleri oluşturuluyor...")
        
        # En önemli 5 sayısal özellik için PDP'ler
        if hasattr(model, 'feature_importances_'):
            # Sayısal özelliklerin önem sıralaması
            importances = model.feature_importances_[numerical_features_idx]
            indices = np.argsort(importances)[::-1]
            
            # En önemli 5 sayısal özellik
            top_features = [numerical_features_idx[i] for i in indices[:5]]
            
            # PDP grafiklerini oluştur
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_partial_dependence(model, X_train, top_features, feature_names=feature_names,
                                   ax=ax, line_kw={"color": "red"})
            fig.suptitle(f'{name} Kısmi Bağımlılık Grafikleri', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f'pdp_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # İkili PDP etkileşimler (ilk iki önemli özellik)
            if len(top_features) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_partial_dependence(model, X_train, [top_features[:2]], feature_names=feature_names,
                                      kind='both', ax=ax, contour_kw={"cmap": "viridis"})
                fig.suptitle(f'{name} İkili Kısmi Bağımlılık Etkileşimi', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(f'pdp_interaction_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"{name} için PDP grafikleri oluşturuldu.")

def main():
    """Ana fonksiyon"""
    # Veriyi yükle
    df = load_maintenance_data()
    
    if df is not None:
        # Veriyi hazırla
        X_train, X_test, X_train_processed, X_test_processed, X_train_df, X_test_df, y_train, y_test, preprocessor = prepare_data(df)
        
        # Orijinal özelliklerin adlarını al
        original_features = list(X_train.columns)
        
        # İşlenmiş özelliklerin adlarını al
        one_hot_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = one_hot_encoder.get_feature_names_out(['Type'])
        processed_feature_names = list(X_train.columns[:-1]) + list(cat_feature_names)
        
        # Sayısal özelliklerin indekslerini belirle
        numerical_features_idx = list(range(len(X_train.columns[:-1])))
        
        # Modelleri eğit
        print("Modeller eğitiliyor...")
        models = train_models(X_train_processed, y_train)
        
        # Modelleri değerlendir
        print("\nModeller değerlendiriliyor...")
        results = evaluate_models(models, X_test_processed, y_test)
        
        # Karmaşıklık matrislerini görselleştir
        print("\nKarmaşıklık matrisleri oluşturuluyor...")
        plot_confusion_matrices(models, X_test_processed, y_test)
        
        # Özellik önem derecelerini görselleştir
        print("\nÖzellik önem dereceleri görselleştiriliyor...")
        plot_feature_importance(models, X_train_processed, processed_feature_names)
        
        # SHAP değerlerini görselleştir
        print("\nSHAP değerleri görselleştiriliyor...")
        plot_shap_values(models, X_train_df, X_test_df)
        
        # PDP grafiklerini görselleştir
        print("\nPDP grafikleri oluşturuluyor...")
        plot_pdp(models, X_train_processed, processed_feature_names, numerical_features_idx)
        
        print("\nTüm görselleştirmeler tamamlandı!")
    else:
        print("Veri yüklenemediği için işlem yapılamadı.")

if __name__ == "__main__":
    main() 