
"""
MUKAVEMET VE GÜVENLİK ANALİZİ MODÜLÜ
Bu modül, dinamik kuvvetler altındaki uzuvların gerilme (stress) ve güvenlik faktörü (FOS) hesaplamalarını yapar.
"""
import numpy as np

# Malzeme Kütüphanesi (Akma Mukavemeti - Yield Strength [MPa])
MATERIALS = {
    "Yapisal Celik (S235)": 235.0,
    "İmalat Celigi (C45)": 430.0,
    "Aluminyum Alasim (6061-T6)": 276.0,
    "Titanyum Alasim (Ti-6Al-4V)": 880.0,
    "Dökme Demir (G40)": 200.0,  # Çekme mukavemeti referans alınmıştır (gevrek)
    "Pirinç (C36000)": 310.0
}

def calculate_stress_sim(force_N, area_mm2):
    """
    Basit Normal Gerilme Hesabı: sigma = F / A
    Return: Stress [MPa]
    """
    if area_mm2 <= 0: return 0.0
    return force_N / area_mm2

def calculate_safety_factor(stress_MPa, yield_MPa):
    """
    Güvenlik Faktörü: n = Sy / sigma
    """
    if stress_MPa <= 1e-6: return 999.0 # Sonsuz güvenli (yüksüz)
    return yield_MPa / stress_MPa

def analyze_cycle_safety(c_data, material_name, area3, area4):
    """
    Tüm çevrim boyunca Biyel ve Sarkaç/Piston için gerilme ve FOS analizi yapar.
    
    Args:
        c_data: Kinematik/Dinamik analiz sonuçları (F12, F23, F34 listeleri dahil).
        material_name: Seçilen malzeme ismi string.
        area3: Biyel kesit alanı (mm2).
        area4: Sarkaç/Piston kesit alanı (mm2).
        
    Returns:
        stress_results: {
            "sigma3": list, "fos3": list,
            "sigma4": list, "fos4": list,
            "min_fos3": float, "min_fos4": float,
            "yield": float
        }
    """
    yield_val = MATERIALS.get(material_name, 235.0)
    
    # Kuvvetleri al (Büyüklükler)
    # Biyel (Link 3): Üzerindeki kuvvetler F23 (A noktasinda) ve F34 (B noktasinda).
    # Basit yaklaşım: Link üzerindeki maks eksenel kuvveti kritik kabul edelim.
    # Genelde bu tür basit analizde iki ucun büyüklüğünün ortalaması veya büyüğü alınır.
    # Biz F23 ve F34'ü biliyoruz.
    
    f23_list = c_data.get('F23', [])
    f34_list = c_data.get('F34', [])
    f14_list = c_data.get('F14', []) # Sarkaç için yer kuvveti ?
    
    n = len(f23_list)
    sigma3_l = []
    fos3_l = []
    sigma4_l = []
    fos4_l = []
    
    for i in range(n):
        # --- BİYEL (Link 3) ---
        # Biyel üzerindeki maksimum kuvvet yükü
        f_biyel = max(f23_list[i], f34_list[i]) 
        sigma3 = calculate_stress_sim(f_biyel, area3)
        fos3 = calculate_safety_factor(sigma3, yield_val)
        
        sigma3_l.append(sigma3)
        fos3_l.append(fos3)
        
        # --- SARKAÇ / ÇIKIŞ (Link 4) ---
        # Sarkaç üzerindeki kuvvetler F34 ve F14
        f_sarkac = max(f34_list[i], f14_list[i])
        sigma4 = calculate_stress_sim(f_sarkac, area4)
        fos4 = calculate_safety_factor(sigma4, yield_val)
        
        sigma4_l.append(sigma4)
        fos4_l.append(fos4)
        
    return {
        "sigma3": sigma3_l, "fos3": fos3_l,
        "sigma4": sigma4_l, "fos4": fos4_l,
        "min_fos3": min(fos3_l) if fos3_l else 0,
        "min_fos4": min(fos4_l) if fos4_l else 0,
        "max_sigma3": max(sigma3_l) if sigma3_l else 0,
        "max_sigma4": max(sigma4_l) if sigma4_l else 0,
        "yield_strength": yield_val
    }
