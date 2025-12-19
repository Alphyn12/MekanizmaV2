import numpy as np
from materials import MATERIALS_DB

def calculate_fatigue_life(sigma_max, sigma_min, material_key, rpm):
    """
    Basquin Denklemi ve Soderberg/Goodman kriterlerini kullanarak yorulma ömrü tahmini.
    Basitleştirilmiş yaklaşım: Sadece gerilme genliği ve Wöhler eğrisi.
    
    Args:
        sigma_max (float): Maksimum Gerilme [MPa]
        sigma_min (float): Minimum Gerilme [MPa]
        material_key (str): Malzeme anahtarı
        rpm (float): Çalışma hızı [devir/dak]
        
    Returns:
        dict: {
            'life_cycles': float or string ("Sonsuz"),
            'life_hours': float or string ("Sonsuz"),
            'sigma_amp': float,
            'sigma_mean': float,
            'risk_level': str (High/Low/None)
        }
    """
    mat = MATERIALS_DB.get(material_key, MATERIALS_DB["Celik_1050"])
    Sut = mat['Sut']
    Se = mat['Se']
    
    # 1. Gerilme Bileşenleri
    sigma_a = abs(sigma_max - sigma_min) / 2.0  # Genlik
    sigma_m = (sigma_max + sigma_min) / 2.0     # Ortalama
    
    # 2. Goodman Düzeltmesi (Basitleştirilmiş: Ortalama gerilme etkisi ihmal edilebilir
    # veya eşdeğer gerilme (S_eq) hesaplanabilir.
    # Şimdilik direkt sigma_a üzerinden Wöhler eğrisi kullanalım, en temiz yöntem bu seviye için)
    
    # Wöhler Eğrisi Parametreleri (Basitleştirilmiş Çelik için)
    # S = a * N^b
    # Nokta 1: (N=10^3, S=0.9*Sut) (Düşük çevrim yorulması sınırı)
    # Nokta 2: (N=10^6, S=Se) (Sürekli mukavemet sınırı)
    
    Sm_10_3 = 0.9 * Sut
    Sm_10_6 = Se
    
    # Log-Log düzlemde eğim (b) ve katsayı (a)
    # log(S) = log(a) + b * log(N)
    # b = (log(S2) - log(S1)) / (log(N2) - log(N1))
    
    log_N1 = 3.0 # 10^3
    log_N2 = 6.0 # 10^6
    
    log_S1 = np.log10(Sm_10_3)
    log_S2 = np.log10(Sm_10_6)
    
    b = (log_S2 - log_S1) / (log_N2 - log_N1)
    # log(S1) = log(a) + b * 3 => log(a) = log(S1) - 3b
    log_a = log_S1 - 3.0 * b
    a = 10**log_a
    
    # 3. Ömür Hesabı
    # sigma_a = a * N^b  =>  N = (sigma_a / a)^(1/b)
    
    stress_to_check = sigma_a
    
    life_cycles = 0
    life_hours = 0
    risk = "Bilinmiyor"
    
    if stress_to_check < Se:
        life_cycles = float('inf')
        risk = "Yok (Sonsuz Ömür)"
    elif stress_to_check > 0.9 * Sut:
        # Çok yüksek gerilme, hemen kırılır (Statik hasar bölgesine yakın)
        life_cycles = 1 # Hemen hemen anında
        risk = "Çok Yüksek (Statik Hasar Riski)"
    else:
        # Sonlu Ömür Bölgesi
        N = (stress_to_check / a) ** (1.0 / b)
        life_cycles = N
        risk = "Orta (Sonlu Ömür)"
        
    # Saate Çevirme
    # Saat = N / (RPM * 60)
    if life_cycles == float('inf'):
        life_hours = float('inf')
    else:
        if rpm > 0:
            life_hours = life_cycles / (rpm * 60.0)
        else:
            life_hours = float('inf') # Hareket yoksa ömür gitmez :)
            
    return {
        'life_cycles': life_cycles,
        'life_hours': life_hours,
        'sigma_amp': sigma_a,
        'sigma_mean': sigma_m,
        'risk_level': risk,
        'wohler_params': {'a': a, 'b': b, 'Se': Se, 'Sut': Sut} # Plotting için
    }
