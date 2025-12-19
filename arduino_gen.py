
"""
ARDUINO CODE GENERATOR (MECHATRONICS MODULE)
Bu modül, kinematik profil datalarından (zaman-konum) çalıştırılabilir Arduino (.ino) kodu üretir.
"""

def generate_arduino_code(theta_array, motor_type, pin_no, rpm, motion_mode):
    """
    Python verilerinden Arduino C++ kodu oluşturur.
    
    Args:
        theta_array (list/array): 0-360 arası açı değerleri (Krank veya Çıkış açısı olarak kullanılabilir).
                                Genelde çıkış uzvu (Link 4) bir servoya bağlanır.
        motor_type (str): "Standart Servo (0-180)", "Sürekli Servo", "Step Motor"
        pin_no (int): Arduino pini.
        rpm (float): İstenen hız.
        motion_mode (str): "Tek Tur", "Döngüsel", "Sarkaç (Sweep)"
        
    Returns:
        str: Tamamlanmış .ino kodu string.
    """
    import numpy as np
    from datetime import datetime
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. PRE-PROCESSING (DATA MAPPING) ---
    # Standart Servo 0-180 arası çalışır. Gelen veri 360 olabilir veya negatif olabilir.
    # Güvenlik için map edelim.
    # Varsayım: Gelen theta_array bir tam periyodu (0-360 crank) kapsayan ÇIKIŞ AÇILARI (theta4) olabilir
    # veya giriş motorunu simüle ediyorsak Krank Açısıdır.
    # Kullanıcı genellikle Output Link (Sarkaç) hareketini bir servoya yaptırmak ister (Animatronik).
    # Veya Krank motorunu sürmek ister (Step Motor).
    
    # Biz burada verilen diziyi (theta_array) Olduğu Gibi Oynatıcı (Player) mantığı kuracağız.
    # Ancak Servo için 0-180 clamp yapılır.
    
    clean_angles = []
    
    # Veri analizi
    min_val = min(theta_array)
    max_val = max(theta_array)
    
    scale_factor = 1.0
    offset = 0.0
    
    warning_comment = ""
    
    if motor_type == "Standart Servo (0-180)":
        # Check range
        if min_val < 0 or max_val > 180:
             # Auto MAP to 0-180 to be safe? Or Clamp?
             # Let's Map nicely constraints.
             # Eger aralik 0-180 icindeyse dokunma. Degilse map et.
             if (max_val - min_val) > 180:
                 warning_comment = "// UYARI: Hareket araligi 180 dereceyi asiyor. 0-180 araligina olceklendi (Compress)."
                 # Map range to 0-180
                 # val -> (val - min) * (180 / (max-min))
                 def mapper(x): return int((x - min_val) * (180.0 / (max_val - min_val)))
             else:
                 # Shift to fit 0-180 if needed (e.g. -45 to +45 -> 45 to 135)
                 # Center it around 90
                 mid = (max_val + min_val) / 2.0
                 shift = 90.0 - mid
                 warning_comment = f"// UYARI: Hareket ortalandi (Shift: {shift:.1f} deg)."
                 def mapper(x): return int(max(0, min(180, x + shift)))
        else:
             mapper = lambda x: int(x)
             
    elif motor_type == "Step Motor":
        # Step motor için açılar adıma çevrilmeli (Örn: 1.8 deg/step -> 200 step/rev)
        # Sürücü kütüphanesi genelde step(n) ister.
        # Bu örnekte AccelStepper mantığı veya basit for loop kullanılabilir.
        # Basitlik için "Hedef Konum (Adım)" dizisi oluşturalım.
        steps_per_rev = 200 * 8 # 1/8 microstepping varsayalım = 1600 step/tur
        mapper = lambda x: int(x * (steps_per_rev / 360.0))
        
    else:
        # Sürekli servo (0: Max Speed CW, 90: Stop, 180: Max Speed CCW)
        # Bu hız profilidir. Konum verisinden hız türetmek gerekir.
        # Şimdilik standart servo mantığı ile ilerleyelim.
        mapper = lambda x: int(x) # Placeholder

    # Apply mapper
    profile_data = [mapper(val) for val in theta_array]
    
    # Array String
    # Arduino belleği kısıtlıdır (2KB RAM for Uno).
    # 360 elemanlı int dizisi = 720 byte. Okay for UNO.
    # PROGMEM kullanmak daha iyidir ama basit int array yapalım.
    
    array_content = ", ".join(map(str, profile_data))
    
    # --- 2. TIMING CALC ---
    # theta_array'in tüm cycle olduğunu varsayıyoruz (bir tur).
    # Bu bir turu RPM hızında tamamlaması gerek.
    # Total Time (min) = 1 / RPM
    # Total Time (ms) = (60 / RPM) * 1000
    # Step Delay (ms) = Total Time / Number of Steps
    
    num_steps = len(theta_array)
    if rpm <= 0: rpm = 1.0
    total_duration_ms = (60.0 / rpm) * 1000.0
    step_delay = max(1, int(total_duration_ms / num_steps))
    
    # --- 3. CODE TEMPLATE CONSTRUCTION ---
    
    # SERVO TEMPLATE
    if "Servo" in motor_type:
        code = f"""/*
  MEKANIZMA ANALIZ PLATFORMU - OTOMATIK KODURETICI
  Proje: Mekatronik Entegrasyon Modulu
  Olusturma Tarihi: {date_str}
  
  Motor Tipi: {motor_type}
  Hedef Hız: {rpm} RPM (Yaklasik)
  Pin: {pin_no}
  Hareket Modu: {motion_mode}
  Veri Sayisi: {num_steps} nokta
*/

#include <Servo.h>

Servo myServo;  // Servo nesnesi olustur
const int servoPin = {pin_no};

// Hareket Profili (Derece)
{warning_comment}
// RAM tasarrufu icin const kullaniyoruz
const int motionProfile[] = {{
  {array_content}
}};

const int DATA_SIZE = sizeof(motionProfile) / sizeof(motionProfile[0]);
const int STEP_DELAY = {step_delay}; // Adimlar arasi bekleme suresi (ms)

void setup() {{
  Serial.begin(9600);
  Serial.println("Mekanizma Baslatiliyor...");
  
  myServo.attach(servoPin);
  
  // Baslangic pozisyonuna git
  myServo.write(motionProfile[0]);
  delay(1000); // 1 saniye bekle
}}

void loop() {{
  // --- HAREKET DONGUSU ---
  """
        if motion_mode == "Tek Tur (One-Shot)":
            code += """
  static bool done = false;
  if (!done) {
    for (int i = 0; i < DATA_SIZE; i++) {
        myServo.write(motionProfile[i]);
        delay(STEP_DELAY);
    }
    done = true; // Dur
    Serial.println("Hareket Tamamlandi.");
  }
"""
        elif motion_mode == "Sarkaç (Sweep)":
             code += """
  // Ileri Git
  for (int i = 0; i < DATA_SIZE; i++) {
      myServo.write(motionProfile[i]);
      delay(STEP_DELAY);
  }
  
  delay(200); // Kisa bekleme
  
  // Geri Don
  for (int i = DATA_SIZE - 1; i >= 0; i--) {
      myServo.write(motionProfile[i]);
      delay(STEP_DELAY);
  }
"""
        else: # Döngüsel Loop
             code += """
  for (int i = 0; i < DATA_SIZE; i++) {
      myServo.write(motionProfile[i]);
      delay(STEP_DELAY);
  }
"""
        code += """
}
"""

    # STEP MOTOR TEMPLATE (Basic)
    elif motor_type == "Step Motor":
        code = f"""/*
  MEKANIZMA ANALIZ PLATFORMU - STEP MOTOR KODU
  Tarih: {date_str}
  Not: Bu kod bloklama (blocking) yontemi ile calisir. 
  Profesyonel kullanim icin AccelStepper kutuphanesi onerilir.
*/

// Step Motor Pinleri (Ornek: A4988 Surucu)
const int stepPin = {pin_no}; 
const int dirPin = {pin_no + 1}; // Varsayilan bir sonraki pin

// Hedef Konumlar (Adim Cinsinden, 1600 step/tur varsayimi)
const int motionProfile[] = {{
  {array_content}
}};

const int DATA_SIZE = sizeof(motionProfile) / sizeof(motionProfile[0]);
const int STEP_DELAY_MICROS = {step_delay * 1000}; // ms -> us

void setup() {{
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(9600);
}}

void loop() {{
   // Basit Konum Takibi (Simulasyon amacli)
   // Gercek uygulamada mevcut konum ile hedef konum farki kadar adim atilmalidir.
   static int currentStep = 0;

   for (int i = 0; i < DATA_SIZE; i++) {{
      int target = motionProfile[i];
      int stepsToDo = target - currentStep;
      
      if (stepsToDo > 0) digitalWrite(dirPin, HIGH); // Yon 1
      else digitalWrite(dirPin, LOW); // Yon 2
      
      stepsToDo = abs(stepsToDo);
      
      for(int s=0; s<stepsToDo; s++) {{
          digitalWrite(stepPin, HIGH);
          delayMicroseconds(500); // Hiz ayari (sabit)
          digitalWrite(stepPin, LOW);
          delayMicroseconds(500);
      }}
      
      currentStep = target;
      // Zamanlama profili (Animasyon hizi icin)
      delay({step_delay}); 
   }}
}}
"""

    return code
