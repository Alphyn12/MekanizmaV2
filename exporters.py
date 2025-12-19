
"""
IMAGERY VE CAD EXPORT MODÜLÜ (`exporters.py`)
Bu modül, mekanizmanın:
1. DXF formatında teknik resim çıktısını (ezdxf kullanarak).
2. GIF formatında animasyon çıktısını (imageio ve matplotlib kullanarak) oluşturur.
"""

import ezdxf
import imageio
import matplotlib.pyplot as plt
import numpy as np
import io
import os

def export_to_dxf(mechanism_data, cycle_data, filename="mekanizma.dxf"):
    """
    Simülasyonun o anki durumunu ve tüm yörüngeyi DXF olarak kaydeder.
    
    Args:
        mechanism_data: O anki pozisyon {"O2": (x,y), "A": (x,y), ...}
        cycle_data: Tam tur veri (yörüngeler için) {"joints": {"B": [(x,y)..], ...}}
    """
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # --- KATMANLAR ---
    doc.layers.add("UZUVLAR", color=5) # Mavi
    doc.layers.add("MAFSALLAR", color=1) # Kırmızı
    doc.layers.add("YORUNGELER", color=9) # Gri
    doc.layers.add("SABITLER", color=7) # Beyaz/Siyah
    
    # --- 1. O ANKI KONUM (UZUVLAR) ---
    points = mechanism_data
    
    # Sıralı bağlantı çiftleri
    # O2-A (Krank), A-B (Biyel). 
    # 4-Bar: B-O4 (Sarkaç), O4-O2 (Zemin).
    # Slider: B-C (Yok, A-B krank-biyel zaten). C piston.
    
    def add_line(p1, p2, layer):
        if p1 and p2:
            msp.add_line(p1, p2, dxfattribs={'layer': layer, 'lineweight': 35}) # 0.35mm

    def add_circle(center, radius, layer):
        if center:
            msp.add_circle(center, radius, dxfattribs={'layer': layer})

    # Koordinatları al
    O2 = points.get('O2')
    A = points.get('A')
    B = points.get('B')
    O4 = points.get('O4')
    C = points.get('C') # Slider Piston
    
    # Çizim
    # Krank
    add_line(O2, A, "UZUVLAR") # O2 is usually (0,0) or fixed. Check kinematics.
    add_line(points.get('O2', (0,0)), points.get('A'), "UZUVLAR") 
    
    # Biyel
    add_line(points.get('A'), points.get('B'), "UZUVLAR")
    
    # Çıkış
    if O4: # 4-Bar
         add_line(points.get('B'), O4, "UZUVLAR")
         add_line(O4, points.get('O2', (0,0)), "SABITLER") # Ground
         
         add_circle(O4, 5, "MAFSALLAR")
    
    if C: # Slider
         # Piston itself
         # Draw a box around C?
         cx, cy = C
         msp.add_lwpolyline([(cx-10, cy-5), (cx+10, cy-5), (cx+10, cy+5), (cx-10, cy+5), (cx-10, cy-5)], dxfattribs={'layer': 'UZUVLAR'})
    
    # Mafsallar
    add_circle(points.get('O2', (0,0)), 5, "MAFSALLAR")
    add_circle(points.get('A'), 5, "MAFSALLAR")
    add_circle(points.get('B'), 5, "MAFSALLAR")

    # --- 2. YÖRÜNGELER ---
    # B Noktasını çizelim (Biyel ucu / Coupler curve)
    if cycle_data and 'joints' in cycle_data and 'B' in cycle_data['joints']:
        traj_B = cycle_data['joints']['B']
        # Convert to list of (x,y) if not none
        valid_pts = [p for p in traj_B if p is not None]
        if valid_pts:
            msp.add_lwpolyline(valid_pts, dxfattribs={'layer': 'YORUNGELER', 'linetype': 'DASHED'})
            
    # Kaydet
    output = io.StringIO()
    doc.write(output)
    # Streamlit download expects bytes usually or string IO.
    # ezdxf write accepts stream. 
    # But wait, ezdxf writes ASCII DXF content.
    return output.getvalue()


def generate_animation_gif(mechanism, L1, L2, L3, L4, theta_range, filename="animasyon.gif"):
    """
    Matplotlib kullanarak animasyon karesi üretir ve GIF yapar.
    
    Args:
        mechanism: Kinematics sınıfı örneği
        theta_range: [0, 5, 10, ... 360] açı listesi
    """
    frames = []
    
    # Setup Plot limits once
    padding = (L2 + L3 + L4) * 1.2
    
    for theta in theta_range:
        pos = mechanism.get_position(theta)
        if not pos: continue
        
        # Create Frame
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        ax.set_xlim(-padding, padding)
        ax.set_ylim(-padding, padding)
        ax.set_aspect('equal')
        ax.axis('off') # Clean look
        
        # Coordinates
        O2 = pos.get('O2', (0,0))
        A = pos['A']
        B = pos['B']
        
        # Draw Links
        # Link 2 (Crank)
        ax.plot([O2[0], A[0]], [O2[1], A[1]], 'b-', linewidth=4)
        # Link 3 (Coupler)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'g-', linewidth=4)
        
        if 'O4' in pos: # Four Bar
            O4 = pos['O4']
            ax.plot([B[0], O4[0]], [B[1], O4[1]], 'r-', linewidth=4)
            ax.plot([O4[0], O2[0]], [O4[1], O2[1]], 'k--', linewidth=1) # Ground
            ax.plot(O4[0], O4[1], 'ko', markersize=8)
        
        if 'C' in pos: # Slider
            C = pos['C']
            # Piston Box
            rect = plt.Rectangle((C[0]-10, C[1]-5), 20, 10, color='r')
            ax.add_patch(rect)
            # Ground Line
            ax.plot([-padding, padding], [0, 0], 'k:', linewidth=1)
            
        # Joints
        ax.plot(O2[0], O2[1], 'ko', markersize=8)
        ax.plot(A[0], A[1], 'ko', markersize=6)
        ax.plot(B[0], B[1], 'ko', markersize=6)
        
        # Render to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        
        # Read image
        image = imageio.v2.imread(buf)
        frames.append(image)
        
        plt.close(fig)
        buf.close()
        
    # Save GIF
    # Since we need to return bytes for button or file path
    # Let's save to temp file or return bytes? 
    # imageio.mimsave expects filename.
    
    imageio.mimsave(filename, frames, fps=15, loop=0)
    return filename
