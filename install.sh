#!/bin/bash

# Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi Kurulum Betiği

echo "🔧 Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi kurulumu başlatılıyor..."

# Python versiyonunu kontrol et
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "📌 Python sürümü: $python_version"

# Sanal ortam oluştur ve etkinleştir
echo "🔨 Sanal ortam oluşturuluyor..."
python -m venv venv
echo "✅ Sanal ortam oluşturuldu!"

# Sanal ortamı aktive et
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS veya Linux için
    source venv/bin/activate
    echo "✅ Sanal ortam aktive edildi: venv"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows için
    source venv/Scripts/activate
    echo "✅ Sanal ortam aktive edildi: venv"
else
    echo "❌ İşletim sistemi tanınamadı. Lütfen sanal ortamı manuel olarak aktive edin."
    exit 1
fi

# Gereklilikleri kur
echo "🔨 Gerekli paketler kuruluyor..."
pip install -r requirements.txt

# Kurulum durumunu kontrol et
if [ $? -eq 0 ]; then
    echo "✅ Kurulum başarıyla tamamlandı!"
    echo ""
    echo "🚀 Kullanım:"
    echo "  1. Sanal ortamı aktifleştirin: source venv/bin/activate (Linux/macOS) veya venv\\Scripts\\activate (Windows)"
    echo "  2. Görselleştirmeler için: python interactive_visualizations.py"
    echo "  3. Model yorumlanabilirliği için: python model_interpretability.py"
    echo "  4. Dashboard için: streamlit run maintenance_dashboard.py"
    echo "  5. Örnek proje için: python sample-project.py"
    echo ""
    echo "📚 Daha fazla bilgi için README.md dosyasını inceleyin."
else
    echo "❌ Kurulum sırasında bir hata oluştu."
    exit 1
fi 