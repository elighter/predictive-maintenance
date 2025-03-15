#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi
Örnek Uygulama

Bu modül, makine bakımı arıza tahmin modelini eğitip değerlendiren,
tüm temel iş akışını gösteren örnek bir uygulamadır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings

# Uyarıları gizle
warnings.filterwarnings("ignore")

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

def load_data():
    """Bakım veri setini yerel dosyadan yükler"""
    try:
        file_path = "data/predictive_maintenance.csv"
        df = pd.read_csv(file_path)
        print(f"Veri seti başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata: {e}")
        return None

def explore_data(df):
    """Veri setinin keşif analizini yapar"""
    print("\n--- Veri Seti Keşif Analizi ---")
    
    # İlk birkaç satırı kontrol et
    print("\nİlk 5 satır:")
    print(df.head())
    
    # Veri seti hakkında genel bilgi
    print("\nVeri seti bilgisi:")
    print(df.info())
    
    # İstatistiksel özetler
    print("\nİstatistiksel özet:")
    print(df.describe())
    
    # Hedef değişken dağılımı
    print("\nHedef değişken dağılımı:")
    print(df['Target'].value_counts(normalize=True) * 100)
    
    # Kategorik değişkenlerin incelenmesi
    print("\nKategorik değişken analizi:")
    for col in ['Type', 'Failure Type']:
        if col in df.columns:
            print(f"\n{col} dağılımı:")
            print(df[col].value_counts())
    
    # Eksik değer kontrolü
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nEksik değerler:")
        print(missing_values[missing_values > 0])
    else:
        print("\nVeri setinde eksik değer bulunmamaktadır.")

def prepare_data(df):
    """Veriyi model için hazırlar"""
    print("\n--- Veri Hazırlama ---")
    
    # Gereksiz sütunları çıkar
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    y = df['Target']
    
    print(f"Özellik sayısı: {X.shape[1]}")
    print(f"Özellikler: {X.columns.tolist()}")
    
    # Kategorik ve sayısal değişkenleri ayır
    categorical_features = ['Type']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"\nKategorik özellikler: {categorical_features}")
    print(f"Sayısal özellikler: {numerical_features}")
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nEğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Önişleme pipeline'ı
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_evaluate_model(X_train, X_test, y_train, y_test, preprocessor):
    """Modeli eğitir ve değerlendirir"""
    print("\n--- Model Eğitimi ve Değerlendirmesi ---")
    
    # Model pipeline'ı oluştur
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Çapraz doğrulama skoru
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"5-katlı çapraz doğrulama doğruluk skorları: {cv_scores}")
    print(f"Ortalama çapraz doğrulama doğruluk skoru: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Modeli eğit
    model.fit(X_train, y_train)
    print("Model eğitildi.")
    
    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Performans metrikleri
    print("\nPerformans Metrikleri:")
    print(f"Doğruluk: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Kesinlik: {precision_score(y_test, y_pred):.4f}")
    print(f"Duyarlılık: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Skoru: {f1_score(y_test, y_pred):.4f}")
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Karmaşıklık matrisini görselleştir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Karmaşıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")
    
    # ROC eğrisini görselleştir
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC eğrisi 'roc_curve.png' olarak kaydedildi.")
    
    return model

def feature_importance(model, X_train):
    """Özellik önem derecelerini görselleştirir"""
    print("\n--- Özellik Önem Dereceleri ---")
    
    # Özellik önem derecelerini al
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['classifier'].feature_importances_
    
    # Önem derecelerine göre sırala
    indices = np.argsort(importances)[::-1]
    
    # En önemli 10 özelliği göster
    print("\nEn önemli 10 özellik:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices][:10], y=[feature_names[i] for i in indices][:10])
    plt.title('Özellik Önem Dereceleri')
    plt.xlabel('Önem Derecesi')
    plt.ylabel('Özellik')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Özellik önem dereceleri 'feature_importance.png' olarak kaydedildi.")

def make_predictions(model, sample_count=5):
    """Örnek tahminler yapar"""
    print("\n--- Örnek Tahminler ---")
    
    # Yeni veri seti yükle (burada örnek olarak test setinden alıyoruz)
    df = load_data()
    if df is None:
        return
    
    # Rastgele örnek seç
    samples = df.sample(sample_count, random_state=42)
    
    # Özellikler
    X_samples = samples.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    
    # Gerçek sınıflar
    y_true = samples['Target'].values
    
    # Tahmin yap
    y_pred = model.predict(X_samples)
    y_pred_proba = model.predict_proba(X_samples)[:, 1]
    
    # Sonuçları göster
    print("\nÖrnek Tahminler:")
    for i, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_pred_proba)):
        print(f"\nÖrnek {i+1}:")
        print(f"Özellikler: {X_samples.iloc[i].to_dict()}")
        print(f"Gerçek Durum: {'Arıza' if true == 1 else 'Normal'}")
        print(f"Tahmin: {'Arıza' if pred == 1 else 'Normal'}")
        print(f"Arıza Olasılığı: {prob:.4f}")
        
        if true == pred:
            print("Sonuç: Doğru tahmin ✓")
        else:
            print("Sonuç: Yanlış tahmin ✗")

def main():
    """Ana fonksiyon"""
    print("=== Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi ===")
    
    # Veri yükleme
    df = load_data()
    if df is None:
        return
    
    # Veri keşfi
    explore_data(df)
    
    # Veri hazırlama
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Model eğitimi ve değerlendirmesi
    model = train_evaluate_model(X_train, X_test, y_train, y_test, preprocessor)
    
    # Özellik önem dereceleri
    feature_importance(model, X_train)
    
    # Örnek tahminler
    make_predictions(model)
    
    print("\n=== Proje Tamamlandı ===")
    print("Tüm sonuçlar ve grafikler başarıyla kaydedildi.")

if __name__ == "__main__":
    main()