@echo off
REM Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi Kurulum Betiği (Windows)

echo 🔧 Makine Öğrenmesi ile Endüstriyel Bakım Tahmin Projesi kurulumu başlatılıyor...

REM Python versiyonunu kontrol et
python --version
if %ERRORLEVEL% neq 0 (
    echo ❌ Python bulunamadı! Lütfen Python 3.8 veya üzerini kurduğunuzdan emin olun.
    exit /b 1
)

REM Sanal ortam oluştur
echo 🔨 Sanal ortam oluşturuluyor...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo ❌ Sanal ortam oluşturulamadı! Python venv modülünün kurulu olduğundan emin olun.
    exit /b 1
)
echo ✅ Sanal ortam oluşturuldu!

REM Sanal ortamı aktive et
echo 🔨 Sanal ortam aktive ediliyor...
call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo ❌ Sanal ortam aktive edilemedi!
    exit /b 1
)
echo ✅ Sanal ortam aktive edildi: venv

REM Gereklilikleri kur
echo 🔨 Gerekli paketler kuruluyor...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ❌ Paket kurulumu sırasında bir hata oluştu!
    exit /b 1
)

echo ✅ Kurulum başarıyla tamamlandı!
echo.
echo 🚀 Kullanım:
echo   1. Sanal ortamı aktifleştirin: venv\Scripts\activate
echo   2. Görselleştirmeler için: python interactive_visualizations.py
echo   3. Model yorumlanabilirliği için: python model_interpretability.py
echo   4. Dashboard için: streamlit run maintenance_dashboard.py
echo   5. Örnek proje için: python sample-project.py
echo.
echo 📚 Daha fazla bilgi için README.md dosyasını inceleyin.

pause 