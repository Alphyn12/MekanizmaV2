
"""
YORULMA ANALIZİ (FATIGUE ANALYSIS) MODÜLÜ
Bu modül, dinamik yükleme altındaki mekanizma uzuvlarının yorulma ömrünü tahmin eder.
Yaklaşım: S-N Eğrisi (Wöhler) ve Basquin Denklemi.
"""
import numpy as np

# Malzeme Kütüphanesi
# Sut: Çekme Dayanımı (Ultimate Tensile Strength) [MPa]
# Se: Sürekli Mukavemet Sınırı (Endurance Limit) [MPa] (Yaklaşık 0.5 * Sut)
# Celik için Se 10^6 çevrimde tanımlanır.
FATIGUE_MATERIALS = {
    "Celik (AISI 1050)": {"Sut": 690.0, "Se": 345.0},
    "Aluminyum (6061-T6)": {"Sut": 310.0, "Se": 95.0}, # Alüminyumda gerçek bir Se yoktur, 5*10^8 aliyoruz
    "Titanyum (Ti-6Al-4V)": {"Sut": 950.0, "Se": 500.0},
    "Dökme Demir (G40)": {"Sut": 276.0, "Se": 110.0},
    "Paslanmaz Celik (304)": {"Sut": 505.0, "Se": 210.0}
}

def calculate_fatigue_life(sigma_max, sigma_min, material_name, rpm):
    """
    S-N Eğrisi yöntemine göre yorulma ömrünü hesaplar.
    
    Args:
        sigma_max (float): Maksimum gerilme [MPa]
        sigma_min (float): Minimum gerilme [MPa]
        material_name (str): Malzeme ismi
        rpm (float): Çalışma hızı [Devir/Dakika]
        
    Returns:
        dict: {
            "N": float (Çevrim veya inf),
            "life_hours": float (Saat),
            "status": str (Risk durumu),
            "sigma_a": float, "sigma_m": float,
            "Se": float, "Sut": float
        }
    """
    props = FATIGUE_MATERIALS.get(material_name, FATIGUE_MATERIALS["Celik (AISI 1050)"])
    Sut = props["Sut"]
    Se = props["Se"]
    
    # 1. Gerilme Parametreleri
    # Stres değerleri mutlak büyüklük olarak değişkense de, 
    # yorulma için işaretli değerlere ihtiyaç vardır (Basma/Çekme).
    # Basit bir analiz için mutlak max kullanılırsa "Tam Değişken" (Fully Reversed) varsayabiliriz.
    # Ancak burada sigma_max ve min parametre olarak geliyor.
    
    # Basitlik için Amplitude (Genlik) odaklı analiz (Goodman düzeltmesi olmadan, S-N doğrudan usage)
    # sigma_a = (sigma_max - sigma_min) / 2
    # Muhafazakar yaklaşım: Goodman kullanmak yerine Max Stres üzerinden S-N bakmak 
    # Veya sadece sigma_amp'yi Se ile karşılaştırmak (Fully Reversed Assumption).
    # Genelde sigma_a kullanılır.
    
    sigma_a = abs(sigma_max - sigma_min) / 2.0
    sigma_m = (sigma_max + sigma_min) / 2.0

    # Goodman Düzeltmesi ile Eşdeğer Gerilme (Equivalent Fully Reversed Stress)
    # Se_req = sigma_a / (1 - sigma_m/Sut)
    # Eğer sigma_m çok yüksekse ömür kısalır.
    # Negatif sigma_m (Basma) genelde ömrü uzatır ama biz conservative formül kullanalım.
    
    sigma_eq = sigma_a
    if sigma_m > 0: # Sadece çekme ortalama gerilmesi zararlıdır
        denominator = (1 - sigma_m / Sut)
        if denominator > 0.05: # Sıfıra bölme koruması
            sigma_eq = sigma_a / denominator
        else:
            sigma_eq = Sut # Çok yüksek ortalama gerilme -> Kirilma
            
    # Eğer sigma_eq hesaplanan max değerden büyükse onu al
    # Genelde basit analiz için sigma_eq yeterlidir.
    
    stress_val = sigma_eq
    
    # 2. Ömür Hesabı (Basquin S-N Model)
    # N = 10^6 için Se
    # N = 10^3 için 0.9 * Sut (Çelik için genelde)
    
    # S = a * N^b
    # log(S) = log(a) + b * log(N)
    
    S1 = 0.9 * Sut # 1000 çevrimdeki mukavemet
    N1 = 1000.0
    
    S2 = Se # 10^6 (veya Al için 5*10^8)
    N2 = 1.0e6
    if "Aluminyum" in material_name: N2 = 5.0e8
    
    # Eğer gerilme Se altındaysa -> Sonsuz Ömür
    if stress_val < Se:
        life_N = float('inf')
        hours = float('inf')
        status = "SONSUZ ÖMÜR (Güvenli)"
        risk = "DÜŞÜK"
    elif stress_val > S1:
        # Çok yüksek gerilme, Low Cycle Fatigue veya Statik Hasar
        life_N = 1.0 # Anlık hasar
        hours = 0.0
        status = "ANLIK HASAR RISKI (Statik)"
        risk = "KRİTİK"
    else:
        # Sonlu Ömür Bölgesi (Finite Life)
        # b = log(S1/S2) / log(N1/N2)
        import math
        b = math.log10(S1 / S2) / math.log10(N1 / N2)
        a = S1 / (N1 ** b)
        
        # S = a * N^b  =>  N = (S/a)^(1/b)
        life_N = (stress_val / a) ** (1.0 / b)
        
        # Süre Hesabı
        if rpm > 0:
            minutes = life_N / rpm
            hours = minutes / 60.0
        else:
            hours = float('inf')
            
        if life_N < 1e5:
            risk = "YÜKSEK"
            status = "KISA ÖMÜR"
        elif life_N < 1e6:
            risk = "ORTA"
            status = "SINIRLI ÖMÜR"
        else:
            risk = "DÜŞÜK"
            status = "UZUN ÖMÜR"

    return {
        "N": life_N,
        "life_hours": hours,
        "status": status,
        "risk": risk,
        "sigma_calc": stress_val,
        "Sut": Sut,
        "Se": Se,
        "S1K": S1 # 1000 cycle strength
    }
