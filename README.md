# 🔧 Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi

Bu proje, makine öğrenmesi teknikleri kullanarak endüstriyel makinelerin arızalarını önceden tahmin etmeyi amaçlayan bir analiz ve modelleme çalışmasıdır. Sensör verileriyle prediktif bakım stratejileri geliştirmek için kapsamlı bir çerçeve sunar.

## 📋 Proje İçeriği

Proje aşağıdaki ana bileşenleri içermektedir:

1. **İnteraktif Görselleştirmeler** (`interactive_visualizations.py`): Plotly kullanarak veri setinin etkileşimli analizleri
2. **Model Yorumlanabilirliği** (`model_interpretability.py`): SHAP değerleri ve Kısmi Bağımlılık Grafikleri (PDP) ile model açıklamaları 
3. **Streamlit Dashboard** (`maintenance_dashboard.py`): Tam kapsamlı, kullanıcı dostu analiz panosu
4. **Örnek Proje** (`sample-project.py`): Temel makine öğrenmesi iş akışını gösteren ana uygulama

## 🚀 Kurulum

Projeyi çalıştırmak için aşağıdaki kütüphaneleri kurmanız gerekmektedir:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit kagglehub shap
```

### Veri Seti

Bu projede kullanılan veri seti, [Kaggle'dan Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) veri setidir. Veri seti otomatik olarak `kagglehub` kütüphanesi aracılığıyla indirilecektir.

## 🛠️ Kullanım

### 1. İnteraktif Görselleştirmeler

İnteraktif Plotly grafiklerini oluşturmak ve HTML dosyaları olarak kaydetmek için:

```bash
python interactive_visualizations.py
```

Bu komut aşağıdaki HTML dosyalarını oluşturacaktır:
- `ariza_dagilimi.html`: Arıza dağılımı pasta grafiği
- `ariza_turleri.html`: Arıza türleri dağılımı bar grafiği
- `3d_sensor_uzayi.html`: 3D sensör verileri dağılım grafiği
- `korelasyon_matrisi.html`: İnteraktif korelasyon matrisi
- `urun_tipleri_ariza.html`: Ürün tiplerine göre arıza oranları
- `sensor_dagilimi.html`: Sensör değerlerinin arıza durumuna göre dağılımı
- `sicaklik_tork_animasyon.html`: Alet aşınmasına göre animasyonlu değişim grafiği

### 2. Model Yorumlanabilirliği

SHAP ve PDP grafikleri oluşturmak için:

```bash
python model_interpretability.py
```

Bu komut PNG formatında çeşitli model yorumlama grafikleri oluşturacaktır:
- Karmaşıklık matrisleri
- Özellik önem dereceleri
- SHAP önem grafikleri
- SHAP özet grafikleri
- SHAP kuvvet grafikleri
- Kısmi bağımlılık grafikleri

### 3. Streamlit Dashboard

İnteraktif dashboard'u çalıştırmak için:

```bash
streamlit run maintenance_dashboard.py
```

Bu komut, aşağıdaki özelliklere sahip bir web uygulaması başlatacaktır:
- Veri analizi görselleştirmeleri
- Dinamik filtreler
- Model eğitimi ve değerlendirmesi
- Sensör değerlerine göre arıza tahmini simülasyonu

### 4. Temel Örnek Proje

Örnekleme projesi, temel iş akışını görmek için:

```bash
python sample-project.py
```

## 📊 Özellikler

### Veri Analizi
- Çok değişkenli sensör verisi korelasyon analizi
- Arıza dağılımı ve türlerinin incelenmesi
- Ürün tipi ve arıza ilişkisi analizi
- 3D sensör uzayında makine davranışı görselleştirmesi

### Modelleme
- Random Forest ve Gradient Boosting sınıflandırıcıları 
- Çapraz doğrulama ve hiperparametre optimizasyonu
- Performans metrikleri ve değerlendirme grafikleri

### Model Yorumlanabilirliği
- SHAP değerleri ile özellik önemleri
- SHAP kuvvet grafikleri ile bireysel tahmin açıklamaları
- Kısmi bağımlılık grafikleri (PDP) ile özellik etkileri

### Dashboard
- İnteraktif veri keşfi
- Gerçek zamanlı model eğitimi
- Kullanıcı girişli tahmin simülasyonu
- Filtreleme ve özellik seçimi

## 📈 Sonuçlar

Bu proje, endüstriyel makineler için prediktif bakım stratejilerinin geliştirilmesinde makine öğrenmesi uygulamasının etkili bir örneğidir. Sonuçlar şunları göstermektedir:

- Makine arızaları yüksek doğrulukla tahmin edilebilir
- Özellikle H tipi makinelerde ve yüksek alet aşınması durumlarında arıza olasılığı artmaktadır
- Sıcaklık farkı ve dönüş hızı kritik izleme parametreleridir
- Veriye dayalı bakım kararları sayesinde kaynaklar daha verimli kullanılabilir

## 🧰 Geliştirme

Projeye katkıda bulunmak için:

1. Fork edin
2. Yeni özellik dalı oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Dalınızı push edin (`git push origin feature/yeni-ozellik`)
5. Pull request oluşturun

## 📝 Lisans

Bu proje [MIT Lisansı](LICENSE) altında dağıtılmaktadır.

## 📧 İletişim

Sorularınız ve önerileriniz için: [emrecakmak@me.com](mailto:emrecakmak@me.com)

---

#### Kaynak ve Referanslar

1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
2. SHAP: A Game Theoretic Approach to Explain the Output of any Machine Learning Model, Lundberg & Lee, NIPS 2017.
3. Kaggle Dataset: "Machine Predictive Maintenance Classification", Shivam Bansal, 2022.
4. McKinsey & Company. "Predictive maintenance: Taking proactive measures based on advanced data analytics to predict and avoid machine failure." 
