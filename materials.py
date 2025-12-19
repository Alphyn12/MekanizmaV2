# Malzeme Veritabanı
# Birimler: MPa (N/mm^2), g/cm^3

MATERIALS_DB = {
    "Celik_1050": {
        "name": "Çelik 1050 (HR)",
        "Sy": 580.0,       # Akma Mukavemeti [MPa]
        "Sut": 900.0,      # Çekme Dayanımı [MPa]
        "Se": 450.0,       # Sürekli Mukavemet Sınırı (Sut * 0.5) [MPa]
        "density": 7.85,   # Yoğunluk [g/cm^3]
        "color": "#34495E" # Koyu Gri/Mavi
    },
    "Aluminyum_6061": {
        "name": "Alüminyum 6061-T6",
        "Sy": 276.0,
        "Sut": 310.0,
        "Se": 95.0,        # Alüminyumda Se yoktur, 5e8 çevrimdeki değerdir
        "density": 2.70,
        "color": "#BDC3C7" # Gümüş
    },
    "Titanyum_Ti6Al4V": {
        "name": "Titanyum Ti-6Al-4V",
        "Sy": 880.0,
        "Sut": 950.0,
        "Se": 500.0,
        "density": 4.43,
        "color": "#8E44AD" # Morumsu
    },
    "ABS_Plastik": {
        "name": "ABS Plastik (Endüstriyel)",
        "Sy": 40.0,
        "Sut": 45.0,
        "Se": 15.0,
        "density": 1.04,
        "color": "#E67E22" # Turuncu
    }
}
