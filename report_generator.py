from fpdf import FPDF
import tempfile
import datetime

def clean_text(text):
    """
    Türkçe karakterleri Latin-1 uyumlu İngilizce karakterlere dönüştürür.
    FPDF'in standart fontuyla çökmeden çalışmasını sağlar.
    """
    if not isinstance(text, str):
        text = str(text)
    
    replacements = {
        'ı': 'i', 'İ': 'I',
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C',
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    
    # FPDF'in encode edebilmesi için latin-1'e zorla, yapamazsa ? koy
    return text.encode('latin-1', 'replace').decode('latin-1')

class PDFReport(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        txt = clean_text('Bu rapor Python & Streamlit altyapısı ile otomatik oluşturulmuştur.')
        self.cell(0, 10, txt, 0, 0, 'C')

def create_table_header(pdf, header_labels, col_widths):
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240) # Light Gray
    pdf.set_text_color(0, 0, 0)
    for i, label in enumerate(header_labels):
        pdf.cell(col_widths[i], 8, clean_text(label), 1, 0, 'C', True)
    pdf.ln()

def create_table_row(pdf, row_data, col_widths):
    pdf.set_font('Arial', '', 10)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(0, 0, 0)
    for i, data in enumerate(row_data):
        text = str(data)
        pdf.cell(col_widths[i], 8, clean_text(text), 1, 0, 'C', False)
    pdf.ln()

def create_pdf(L1, L2, L3, L4, omega2, alpha2, mech_type, theta2_inst, inst_results, cycle_stats):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False, margin=0)
    
    # --- HEADER ---
    pdf.set_font('Arial', 'B', 14)
    # Dynamic Title
    title = 'DORT CUBUK MEKANIZMASI ANALIZ RAPORU' if "4" in str(mech_type) or "Dört" in str(mech_type) else 'KRANK-BIYEL MEKANIZMASI ANALIZ RAPORU'
    pdf.cell(0, 10, clean_text(title), 0, 1, 'C')
    
    pdf.set_font('Arial', 'I', 9)
    current_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.cell(0, 5, clean_text(f'Rapor Tarihi: {current_time}'), 0, 1, 'R')
    
    # Thick Horizontal Line
    pdf.set_line_width(0.5)
    pdf.line(10, 25, 200, 25)
    pdf.set_line_width(0.2) # Reset to thin
    pdf.ln(10)
    
    # --- SECTION 1: SYSTEM PARAMETERS ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, clean_text('1. SİSTEM PARAMETRELERİ'), 0, 1, 'L')
    pdf.ln(2)
    
    col_w_params = [45, 45, 45, 55] 
    
    create_table_header(pdf, ['Parametre', 'Değer', 'Parametre', 'Değer'], col_w_params)
    
    create_table_row(pdf, [
        'Sabit Uzuv (L1)', f'{L1} mm', 
        'Giriş Hızı (w2)', f'{omega2} rad/s'
    ], col_w_params)
    
    create_table_row(pdf, [
        'Krank (L2)', f'{L2} mm', 
        'Giriş İvmesi (a2)', f'{alpha2} rad/s^2'
    ], col_w_params)
    
    mech_name_short = str(mech_type)[0:25]
    create_table_row(pdf, [
        'Biyel (L3)', f'{L3} mm', 
        'Mekanizma Tipi', mech_name_short
    ], col_w_params)
    
    create_table_row(pdf, [
        'Sarkaç (L4)', f'{L4} mm', 
        '', ''
    ], col_w_params)
    
    pdf.ln(10)
    
    # --- SECTION 2: CYCLE PERFORMANCE ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, clean_text('2. ÇEVRİM BOYUNCA EKSTREMLER (0-360 DERECE)'), 0, 1, 'L')
    pdf.ln(2)
    
    col_w_cycle = [80, 50, 60]
    
    is_slider = "Krank" in str(mech_type)
    
    create_table_header(pdf, ['Veri Tipi', 'Maksimum Değer', 'Gerçekleştiği Açı / Aralık'], col_w_cycle)
    
    # Safe getters for stats
    def get_stat(key, fmt_str="{:.2f}"):
        val = cycle_stats.get(key)
        if val is None: return "N/A"
        try:
            return fmt_str.format(val)
        except:
            return str(val)

    # 1. Biyel Speed
    create_table_row(pdf, [
        'Maks. Biyel Hızı (w3)', 
        f"{get_stat('max_w3')} rad/s", 
        f"@ {get_stat('max_w3_angle', '{:.1f}')} deg"
    ], col_w_cycle)
    
    # 2. Output Speed
    if not is_slider:
        create_table_row(pdf, [
            'Maks. Çıkış Hızı (w4)', 
            f"{get_stat('max_w4')} rad/s", 
            f"@ {get_stat('max_w4_angle', '{:.1f}')} deg"
        ], col_w_cycle)
    else:
         create_table_row(pdf, [
            'Maks. Piston Hızı (Vp)', 
            f"{get_stat('max_v_piston')} mm/s", 
            f"@ {get_stat('max_v_piston_angle', '{:.1f}')} deg"
        ], col_w_cycle)

    # 3. Biyel Accel
    create_table_row(pdf, [
        'Maks. Biyel İvmesi (alpha3)', 
        f"{get_stat('max_alpha3')} rad/s^2", 
        f"@ {get_stat('max_alpha3_angle', '{:.1f}')} deg"
    ], col_w_cycle)
    
    # 4. Output Accel
    if not is_slider:
        create_table_row(pdf, [
            'Maks. Çıkış İvmesi (alpha4)', 
            f"{get_stat('max_a4')} rad/s^2", 
            f"@ {get_stat('max_a4_angle', '{:.1f}')} deg"
        ], col_w_cycle)
    else:
        create_table_row(pdf, [
            'Maks. Piston İvmesi (Ap)', 
            f"{get_stat('max_a_piston')} mm/s^2", 
            f"@ {get_stat('max_a_piston_angle', '{:.1f}')} deg"
        ], col_w_cycle)

    # 5. Transmission Angle (Only 4-Bar)
    if not is_slider:
        min_mu = get_stat('min_mu', '{:.1f}')
        max_mu = get_stat('max_mu', '{:.1f}')
        create_table_row(pdf, [
            'Bağlama Açısı Aralığı (mu)', 
            f"{min_mu} - {max_mu} deg", 
            "Tüm Çevrim"
        ], col_w_cycle)
        
    pdf.ln(10)
    
    # --- SECTION 3: INSTANT ANALYSIS ---
    pdf.set_font('Arial', 'B', 12)
    label_sec3 = f'3. SEÇİLEN AÇIDAKİ DETAYLI ANALİZ (Theta2 = {theta2_inst} deg)'
    pdf.cell(0, 8, clean_text(label_sec3), 0, 1, 'L')
    pdf.ln(2)
    
    col_w_inst = [95, 95]
    create_table_header(pdf, ['Parametre', 'Değer'], col_w_inst)
    
    # Use .get() for results too
    w3_i = inst_results.get('w3', 0)
    a3_i = inst_results.get('a3', 0)
    w4_i = inst_results.get('w4', 0) # w4 or Vp
    a4_i = inst_results.get('a4', 0) # a4 or Ap
    mu_i = inst_results.get('mu', 0)
    
    create_table_row(pdf, ['Biyel Hızı (w3)', f"{w3_i:.2f} rad/s"], col_w_inst)
    
    if not is_slider:
        create_table_row(pdf, ['Çıkış Hızı (w4)', f"{w4_i:.2f} rad/s"], col_w_inst)
    else:
        create_table_row(pdf, ['Piston Hızı (Vp)', f"{w4_i:.2f} mm/s"], col_w_inst)
        
    create_table_row(pdf, ['Biyel İvmesi (alpha3)', f"{a3_i:.2f} rad/s^2"], col_w_inst)
    
    if not is_slider:
        create_table_row(pdf, ['Çıkış İvmesi (alpha4)', f"{a4_i:.2f} rad/s^2"], col_w_inst)
        create_table_row(pdf, ['Bağlama Açısı (mu)', f"{mu_i:.2f} deg"], col_w_inst)
    else:
        create_table_row(pdf, ['Piston İvmesi (Ap)', f"{a4_i:.2f} mm/s^2"], col_w_inst)
        
    filename = f"kinematik_analiz_raporu_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def generate_matlab_code(cycle_data, params, mech_type):
    """
    Generates a MATLAB R2015 compatible .m script for mechanism simulation.
    Features:
    - Legacy syntax (single quotes, set() for properties).
    - Struct based data storage.
    - Animation loop.
    - Export to Excel.
    """
    import pandas as pd
    import numpy as np
    
    # helper to formatting arrays to Matlab
    # helper to formatting arrays to Matlab
    def to_ml_arr(arr):
        if isinstance(arr, pd.Series): arr = arr.values
        if isinstance(arr, (list, tuple, np.ndarray)):
            flat = np.array(arr, dtype=object).flatten() # Ensure object dtype to keep Nones
            res = "["
            for i, v in enumerate(flat):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    res += "NaN"
                else:
                    try:
                        res += f"{float(v):.4f}"
                    except:
                        res += "NaN"
                        
                if i < len(flat) - 1:
                    res += ", "
                    if (i + 1) % 20 == 0:
                        res += " ...\n"
            res += "]"
            return res
        return "[]"

    # 1. PARAMETERS STRUCT
    params_str = ""
    for k, v in params.items():
        if isinstance(v, (int, float)):
            params_str += f"params.{k} = {v};\n"
        elif isinstance(v, str):
            params_str += f"params.{k} = '{v}';\n"
            
    # 2. DATA ARRAYS
    data_block = ""
    valid_keys = ['theta2', 'theta3', 'theta4', 'omega3', 'output_vel', 'alpha3', 'output_acc', 'mu', 'T_input']
    
    # Process Coordinates
    joints = cycle_data.get('joints', {})
    coords_block = ""
    
    # Coordinates
    if 'A' in joints:
        Ax = [p[0] for p in joints['A']]
        Ay = [p[1] for p in joints['A']]
        coords_block += f"data.Ax = {to_ml_arr(Ax)};\n"
        coords_block += f"data.Ay = {to_ml_arr(Ay)};\n"
        
    if 'B' in joints:
        Bx = [p[0] for p in joints['B']]
        By = [p[1] for p in joints['B']]
        coords_block += f"data.Bx = {to_ml_arr(Bx)};\n"
        coords_block += f"data.By = {to_ml_arr(By)};\n"
        
    if str(mech_type) == "Krank-Biyel Mekanizması": # Correct string check
        # Slider Crank used 'B' as Piston Pin in Kinematics cycle?
        # Check kinematics.py analyze_cycle: B is piston pin. So Bx, By is enough.
        # But we might want 'Cx' if specifically needed. 
        # Usually B is Piston Pin.
        pass
    
    for k in valid_keys:
        if k in cycle_data and cycle_data[k] is not None:
            m_name = k.replace("output_vel", "w4").replace("output_acc", "alpha4").replace("T_input", "Torque")
            val = cycle_data[k]
            data_block += f"data.{m_name} = {to_ml_arr(val)};\n"

    # 3. MATLAB SCRIPT
    script = f"""% MEKANIZMA ANALIZI - MATLAB SCRIPT
% Otomatik olustulmustur.
% Tarih: {datetime.datetime.now().strftime("%d.%m.%Y %H:%M")}
% Uyumluluk: MATLAB R2015 ve uzeri

clear all; close all; clc;

%% 1. PARAMETRELER
{params_str}

%% 2. ANALIZ VERILERI
{coords_block}
{data_block}

%% 3. RAPORLAMA (Command Window)
fprintf('--- ANALIZ SONUCLARI ---\\n');
if isfield(data, 'Torque')
    fprintf('Maksimum Tork: %.2f Nm\\n', max(abs(data.Torque)));
end
if isfield(data, 'w4')
    fprintf('Maksimum Cikis Hizi: %.2f rad/s\\n', max(abs(data.w4)));
end

%% 4. EXCEL AKTARIM
% R2015 uyumlu table ve writetable
try
    % Struct alanlarini cell array'e cevirip tablo yapalim
    fNames = fieldnames(data);
    T = table();
    for i = 1:length(fNames)
        fVal = data.(fNames{{i}});
        if isnumeric(fVal) && length(fVal) > 1
             colVal = fVal(:);
             % Ensure same length
             if height(T) == 0 || height(T) == length(colVal)
                T.(fNames{{i}}) = colVal;
             end
        end
    end
    
    writetable(T, 'Analiz_Sonuclari.xlsx');
    fprintf('Veriler Analiz_Sonuclari.xlsx dosyasina kaydedildi.\\n');
catch err
    fprintf('Excel hatasi: %s\\n', err.message);
end

%% 5. LIVE ANIMASYON (R2015 Uyumlu - set metodu)
fprintf('Animasyon baslatiliyor...\\n');
figure('Name', 'Mekanizma Simulasyonu', 'NumberTitle', 'off', 'Color', 'w');
grid on; axis equal; hold on;
xlabel('X (mm)'); ylabel('Y (mm)');

% Cizim Sinirlari
pad = (params.L2 + params.L3) * 1.5;
axis([-pad pad -pad pad]);

% Grafik Nesneleri (Handles)
plot(0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k'); % O2

h_crank = plot([0,0], [0,0], 'b-', 'LineWidth', 3); % Mavi
h_coupler = plot([0,0], [0,0], 'g-', 'LineWidth', 3); % Yesil
h_output = plot([0,0], [0,0], 'r-', 'LineWidth', 3); % Kirmizi/Piston

title('Mekanizma Hareketi');

% Animasyon Dongusu
if isfield(data, 'Ax')
    n = length(data.Ax);
    for k = 1:2:n
        Ax = data.Ax(k); Ay = data.Ay(k);
        Bx = data.Bx(k); By = data.By(k);
        
        % Update Crank
        set(h_crank, 'XData', [0, Ax], 'YData', [0, Ay]);
        
        % Update Coupler
        set(h_coupler, 'XData', [Ax, Bx], 'YData', [Ay, By]);
        
        % Update Output
        if isfield(params, 'L1')
             % 4-Bar
             O4x = params.L1; O4y = 0;
             set(h_output, 'XData', [Bx, O4x], 'YData', [By, O4y]);
             plot(O4x, O4y, 'ko', 'MarkerSize', 5);
        else
             % Slider
             set(h_output, 'XData', [Bx-15, Bx+15], 'YData', [By, By]);
        end
        
        drawnow;
    end
end
fprintf('Animasyon tamamlandi.\\n');
"""
    return script

def generate_excel_report(c_data, params):
    """
    Generates an Advanced Excel Dashboard using xlsxwriter.
    Includes: Conditional Formatting, Sparklines, Formulas, Print Layout.
    """
    import pandas as pd
    import xlsxwriter
    import io

    # Clean Data Setup for DataFrame
    # c_data keys: theta2, T_input, F12, F34 etc.
    # Convert lists to Series
    df_clean = pd.DataFrame()
    df_clean['Aci (deg)'] = pd.Series(c_data.get('theta2', []))
    
    # Tork
    t_in = c_data.get('T_input', [])
    # If it's a list with Nones or mixed, handle it
    # c_data values are usually lists of floats from app.py logic
    df_clean['Tork (Nm)'] = pd.Series(t_in).fillna(0)
    
    # Forces
    df_clean['F_O2_F12 (N)'] = pd.Series(c_data.get('F12', [])).fillna(0)
    df_clean['F_B_F34 (N)'] = pd.Series(c_data.get('F34', [])).fillna(0) # Assuming O4/B
    if 'output_vel' in c_data:
        df_clean['Cikis Hizi (rad/s)'] = pd.Series(c_data.get('output_vel', [])).fillna(0)

    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # FORMATS
    fmt_header = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D7E4BC', 'border': 1})
    fmt_num = workbook.add_format({'num_format': '0.00'})
    fmt_center = workbook.add_format({'align': 'center'})
    fmt_title = workbook.add_format({'bold': True, 'size': 14, 'align': 'center', 'bg_color': '#f0f0f0'})
    
    # --- SHEET 2: DETAYLI VERILER ---
    ws_data = workbook.add_worksheet('Detayli_Veriler')
    
    # Headers
    cols = df_clean.columns.tolist()
    for col_num, value in enumerate(cols):
        ws_data.write(0, col_num, value, fmt_header)
        ws_data.set_column(col_num, col_num, 15)
        
    # Data
    data_len = len(df_clean)
    for i, row in df_clean.iterrows():
        ws_data.write_row(i+1, 0, row.values, fmt_num)
        
    # Conditional Formatting (3-Color Scale)
    # Tork (Col B, index 1), F_O2 (Col C, index 2), F_B (Col D, index 3)
    # Range is 1 to data_len (0-indexed 1 is Row 2)
    
    # Tork
    ws_data.conditional_format(1, 1, data_len, 1, {'type': '3_color_scale'})
    # F_O2
    ws_data.conditional_format(1, 2, data_len, 2, {'type': '3_color_scale'})
    # F_B
    ws_data.conditional_format(1, 3, data_len, 3, {'type': '3_color_scale'})
    
    ws_data.freeze_panes(1, 0)
    
    # --- SHEET 1: OZET RAPOR (Dashboard) ---
    ws_sum = workbook.add_worksheet('Ozet_Rapor')
    ws_sum.set_paper(9) # A4
    ws_sum.set_landscape()
    ws_sum.fit_to_pages(1, 0)
    ws_sum.set_header('&C&14Mekanizma Analiz Raporu')
    ws_sum.set_footer('&C&10Sayfa &P')
    
    ws_sum.set_column('A:A', 25)
    ws_sum.set_column('B:B', 15)
    ws_sum.set_column('C:C', 20)
    ws_sum.set_column('D:D', 30)
    
    ws_sum.merge_range('A1:D1', "ETKİLEŞİMLİ MÜHENDİSLİK PANELİ (EXCEL)", fmt_title)
    
    # Row 3: Torque
    ws_sum.write('A3', "Maksimum Giriş Torku:", fmt_header)
    # Formula: =MAX(Detayli_Veriler!B:B)
    ws_sum.write_formula('B3', f'=MAX(Detayli_Veriler!B2:B{data_len+1})', fmt_num)
    
    ws_sum.write('C3', "Tork Değişimi (Grafik):", fmt_center)
    ws_sum.add_sparkline('D3', {
        'range': f'Detayli_Veriler!B2:B{data_len+1}',
        'markers': True,
        'high_point': True, 'low_point': True,
        'line_weight': 2,
        'series_color': 'blue'
    })
    
    # Row 4: Average
    ws_sum.write('A4', "Ortalama Tork:", fmt_header)
    ws_sum.write_formula('B4', f'=AVERAGE(Detayli_Veriler!B2:B{data_len+1})', fmt_num)
    
    # Row 6: Force
    ws_sum.write('A6', "Maksimum Mafsal Kuvveti:", fmt_header)
    ws_sum.write_formula('B6', f'=MAX(Detayli_Veriler!C2:C{data_len+1})', fmt_num)
    
    ws_sum.write('C6', "Kuvvet Değişimi:", fmt_center)
    ws_sum.add_sparkline('D6', {
        'range': f'Detayli_Veriler!C2:C{data_len+1}',
        'markers': True,
        'high_point': True, 'low_point': True,
        'series_color': 'red'
    })
    
    # Instructions
    ws_sum.merge_range('A8:D8', "NOT: Detayli_Veriler sayfasındaki değerleri değiştirirseniz, bu özet otomatik güncellenir.", workbook.add_format({'italic': True, 'font_color': 'gray'}))
    
    workbook.close()
    return output.getvalue()

