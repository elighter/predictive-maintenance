# Makine Bakımı Tahmini için Örnek Proje
# Python 3.8+ ile test edilmiştir

# Gerekli kütüphaneleri içe aktaralım
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Makine öğrenmesi kütüphaneleri
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.inspection import permutation_importance
import shap

# Veri yükleme fonksiyonu
def load_maintenance_data(file_path=None):
    """
    Bakım verisini yükler. Eğer dosya yolu belirtilmezse sentetik veri oluşturur.
    """
    if file_path:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Veri yüklenirken hata oluştu: {e}")
            print("Sentetik veri oluşturulacak...")
    
    # Sentetik veri oluştur
    np.random.seed(42)
    n_samples = 10000
    
    # UDI ve Product ID
    udi = np.arange(1, n_samples + 1)
    product_ids = [f"P_{np.random.randint(1, 11)}" for _ in range(n_samples)]
    
    # Ürün tipi (L, M, H)
    types = np.random.choice(['L', 'M', 'H'], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Sensör verileri
    air_temp = np.random.normal(300, 5, n_samples)  # Ortalama 300K, std 5K
    process_temp = air_temp + np.random.normal(15, 3, n_samples)  # İşlem sıcaklığı genelde hava sıcaklığından yüksek
    
    # Dönüş hızı (rpm) - Tipine bağlı olarak farklı dağılımlar
    rotational_speed = np.zeros(n_samples)
    for i, t in enumerate(types):
        if t == 'L':
            rotational_speed[i] = np.random.normal(1500, 100, 1)
        elif t == 'M':
            rotational_speed[i] = np.random.normal(2000, 150, 1)
        else:  # H tipi
            rotational_speed[i] = np.random.normal(2500, 200, 1)
    
    # Tork değerleri
    torque = np.random.normal(40, 10, n_samples) + (types == 'H') * 10
    
    # Alet aşınması - Dönüş hızı ve tork ile ilişkili
    tool_wear = np.random.exponential(50, n_samples) + rotational_speed / 50 + torque / 2
    
    # Arıza olasılığı faktörleri
    failure_prob = (
        0.01 +  # Temel arıza olasılığı
        0.1 * (tool_wear > 200) +  # Alet aşınması etkisi
        0.05 * (process_temp > 320) +  # Yüksek işlem sıcaklığı etkisi
        0.03 * (rotational_speed > 2400) +  # Yüksek dönüş hızı etkisi
        0.02 * (types == 'H')  # H tipi ürün etkisi
    )
    
    # Arıza durumu
    machine_failure = np.random.binomial(1, failure_prob)
    
    # Arıza türleri
    twf = np.zeros(n_samples, dtype=int)  # Tool wear failure
    hdf = np.zeros(n_samples, dtype=int)  # Heat dissipation failure
    pwf = np.zeros(n_samples, dtype=int)  # Power failure
    osf = np.zeros(n_samples, dtype=int)  # Overstrain failure
    rnf = np.zeros(n_samples, dtype=int)  # Random failure
    
    # Arıza türlerini belirle
    for i in range(n_samples):
        if machine_failure[i] == 1:
            failure_type = np.random.choice(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            if failure_type == 'TWF':
                twf[i] = 1
            elif failure_type == 'HDF':
                hdf[i] = 1
            elif failure_type == 'PWF':
                pwf[i] = 1
            elif failure_type == 'OSF':
                osf[i] = 1
            else:  # RNF
                rnf[i] = 1
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'UDI': udi,
        'Product ID': product_ids,
        'Type': types,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Machine failure': machine_failure,
        'TWF': twf,
        'HDF': hdf,
        'PWF': pwf,
        'OSF': osf,
        'RNF': rnf
    })
    
    return df

# Veri keşfi ve görselleştirme fonksiyonları
def explore_data(df):
    """
    Veri setini keşfeder ve temel istatistikleri görüntüler
    """
    print(f"Veri seti boyutu: {df.shape}")
    print("\nVeri seti örneği:")
    print(df.head())
    
    print("\nVeri türleri:")
    print(df.dtypes)
    
    print("\nÖzet istatistikler:")
    print(df.describe())
    
    print("\nEksik değerler:")
    print(df.isnull().sum())
    
    print("\nArıza dağılımı:")
    print(df['Machine failure'].value_counts())
    print(f"Arıza oranı: %{df['Machine failure'].mean() * 100:.2f}")
    
    print("\nArıza türleri dağılımı:")
    failure_types = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()
    print(failure_types)
    
    print("\nÜrün tiplerine göre arıza oranları:")
    type_failure = df.groupby('Type')['Machine failure'].mean() * 100
    for t, rate in type_failure.items():
        print(f"{t} tipi: %{rate:.2f}")

def plot_data_distributions(df):
    """
    Veri dağılımlarını görselleştirir
    """
    # Sayısal değişkenlerin dağılımları
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue='Machine failure', kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Dağılımı')
        axes[i].grid(True)
    
    # Arıza dağılımı
    failure_counts = df['Machine failure'].value_counts()
    axes[5].pie(failure_counts, labels=['Normal', 'Arıza'], autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    axes[5].set_title('Makine Arıza Dağılımı')
    
    plt.tight_layout()
    plt.show()
    
    # Arıza türleri dağılımı
    failure_types = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()
    plt.figure(figsize=(10, 6))
    ax = failure_types.plot(kind='bar', color='darkred')
    ax.set_title('Arıza Türleri Dağılımı')
    ax.set_xlabel('Arıza Türü')
    ax.set_ylabel('Sayı')
    for i, v in enumerate(failure_types):
        ax.text(i, v + 0.1, str(v), ha='center')
    plt.tight_layout()
    plt.show()
    
    # Ürün tiplerine göre arıza oranları
    plt.figure(figsize=(10, 6))
    type_failure = df.groupby('Type')['Machine failure'].mean() * 100
    ax = type_failure.plot(kind='bar', color='darkblue')
    ax.set_title('Ürün Tiplerine Göre Arıza Oranları')
    ax.set_xlabel('Ürün Tipi')
    ax.set_ylabel('Arıza Oranı (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(type_failure):
        ax.text(i, v + 1, f"%{v:.2f}", ha='center')
    plt.tight_layout()
    plt.show()

def plot_correlations(df):
    """
    Değişkenler arasındaki korelasyonları görselleştirir
    """
    # Sayısal değişkenlerin korelasyon matrisi
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                     'Machine failure']
    
    corr = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, center=0, square=True, linewidths=.5)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.show()
    
    # Sıcaklık ve arıza ilişkisi
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Air temperature [K]', y='Process temperature [K]', 
                   hue='Machine failure', size='Tool wear [min]', sizes=(20, 200))
    plt.title('Hava Sıcaklığı, İşlem Sıcaklığı ve Arıza İlişkisi')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Dönüş hızı ve tork ilişkisi
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Torque [Nm]', 
                   hue='Machine failure', style='Type')
    plt.title('Dönüş Hızı, Tork ve Arıza İlişkisi')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def feature_engineering(df):
    """
    Veri setine yeni özellikler ekler
    """
    # Kopya oluşturalım
    df_new = df.copy()
    
    # Yeni özellikler ekleyelim
    # 1. Sıcaklık farkı
    df_new['temp_difference'] = df_new['Process temperature [K]'] - df_new['Air temperature [K]']
    
    # 2. Güç (dönüş hızı * tork)
    df_new['power'] = df_new['Rotational speed [rpm]'] * df_new['Torque [Nm]'] / 1000  # kW cinsinden
    
    # 3. Verimlilik oranı
    df_new['efficiency_ratio'] = df