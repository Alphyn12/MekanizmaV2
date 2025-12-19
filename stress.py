import numpy as np
from materials import MATERIALS_DB

def calculate_stress_safety(forces, area_mm2, material_key):
    """
    Hesaplar:
    1. Normal Gerilme (sigma = F / A)
    2. Güvenlik Faktörü (n = Sy / sigma_max)
    
    Args:
        forces (list/array): Kuvvet değerleri [N] (Döngü boyunca)
        area_mm2 (float): Kesit Alanı [mm^2]
        material_key (str): MATERIALS_DB anahtarı
        
    Returns:
        dict: {
            'sigma': [list of stress values MPa],
            'sigma_max': float,
            'sigma_min': float,
            'fos': float (Safety Factor),
            'material': str,
            'is_safe': bool
        }
    """
    if not forces or area_mm2 <= 0:
        return None
        
    mat = MATERIALS_DB.get(material_key, MATERIALS_DB["Celik_1050"])
    Sy = mat['Sy']
    
    # Gerilme Hesabı (MPa) = N / mm^2
    # Kuvvetlerin mutlak değerini mi yoksa işaretli mi almalıyız?
    # Mukavemet için en kritik durum genellikle Max Çeki veya Max Bası'dır.
    # Sünek malzemelerde (Max Shear Stress Teorisini basitleştirerek) Von Mises ~ |Sigma| alabiliriz.
    # Şimdilik basitçe sigma = F/A diyelim.
    
    forces_arr = np.array(forces)
    sigma_arr = forces_arr / area_mm2
    
    # Kritik değerler (Mutlak max gerilme hasar kriteridir)
    abs_sigma = np.abs(sigma_arr)
    sigma_max_abs = np.max(abs_sigma)
    
    if sigma_max_abs < 1e-9:
        n = 999.0 # Sonsuz güvenli
    else:
        n = Sy / sigma_max_abs
        
    return {
        'sigma': sigma_arr.tolist(),
        'sigma_max': np.max(sigma_arr),
        'sigma_min': np.min(sigma_arr),
        'sigma_max_abs': sigma_max_abs,
        'fos': n,
        'material': mat['name'],
        'Sy': Sy,
        'is_safe': n >= 1.0
    }
