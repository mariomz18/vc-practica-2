import cv2
import numpy as np
import time
import os

def alinear_canales(img_ref, img_mov):
    # Extraemos solo el centro de la imagen para evitar que los bordes dañados de las placas de cristal arruinen la correlación
    h, w = img_ref.shape
    cx, cy = w // 2, h // 2
    d = min(w, h) // 4  # Usamos una ventana central
    
    ref_centro = img_ref[cy - d:cy + d, cx - d:cx + d]
    mover_centro = img_mov[cy - d:cy + d, cx - d:cx + d]

    bordes_ref = cv2.Canny(ref_centro, 50, 150)
    bordes_mov = cv2.Canny(mover_centro, 50, 150)


    # Correlación de fase (Fourier)
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(bordes_mov), np.float32(bordes_ref))

    # Otras correlaciones
    # Ahora mismo no van correctamente, hsy que modificar el return

    #(Plantilla més petita)
    #m = 40
    #plantilla_bordes = bordes_ref[m:-m, m:-m]
    
    # 2. Correlació Normalitzada (NCC)
    #res_ncc = cv2.matchTemplate(borde_mov, plantilla_bordes, cv2.TM_CCOEFF_NORMED)
    #_, _, _, max_loc = cv2.minMaxLoc(res_ncc)

    #dx = m - max_loc[0]
    #dy = m - max_loc[1]

    # 3. Suma de Diferències al Quadrat (SSD)
    #res_ssd = cv2.matchTemplate(bordes_mov, plantilla_bordes, cv2.TM_SQDIFF)
    #min_val, _, min_loc, _ = cv2.minMaxLoc(res_ssd)

    #dx = m - min_loc[0]
    #dy = m - min_loc[1]

    return int(round(dx)), int(round(dy))


def mejoras_extra(imagen_bgr): 
    # 1. Eliminación de bordes y marcos (Recortamos un 6% de los márgenes)
    alto, ancho = imagen_bgr.shape[:2]
    margen_y = int(alto * 0.06)
    margen_x = int(ancho * 0.06)
    img_final = imagen_bgr[margen_y:alto-margen_y, margen_x:ancho-margen_x]

    # 2. Eliminación de defectos preservando los bordes (Filtro Bilateral)
    # Parámetros: diámetro=9, sigmaColor=75, sigmaSpace=75
    # img_limpia = cv2.bilateralFilter(img_recortada, 9, 75, 75)

    # 3. Corrección fotométrica sin alterar los colores (CLAHE en el espacio LAB)
    # Convertimos la imagen del espacio BGR al espacio LAB
    # lab = cv2.cvtColor(img_limpia, cv2.COLOR_BGR2LAB)
    
    # Separamos los canales: L (Luminosidad/Brillo), A y B (Componentes de color)
    # l, a, b = cv2.split(lab)
    
    # Creamos y aplicamos el ecualizador CLAHE ÚNICAMENTE al canal L
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l_clahe = clahe.apply(l)
    
    # Volvemos a unir los canales (la luminosidad mejorada + los colores originales)
    # lab_mejorado = cv2.merge([l_clahe, a, b])
    
    # Convertimos la imagen de vuelta al espacio BGR habitual de OpenCV
    #img_final = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)

    return img_final


def procesar_imagen(ruta_imagen):
    # Tasca 1 - Escollir el dataset / Lectura de la imagen
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    print(f"\nTrabajando con: {ruta_imagen}")

    t0 = time.time() # Calculo del tiempo (Se pide)

    # Tasca 3 - Implementar la solucio.

    # 3.1. Recorte de subimágenes
    # Vamos a divir la altura total entre 3 (B, G, R).
    alto_canal = img.shape[0] // 3
    B, G, R = img[0:alto_canal], img[alto_canal:2*alto_canal], img[2*alto_canal:3*alto_canal]

    # 3.2. Puesta en correspondencia (Registrado)
    # Calculamos desplazamientos usando B como referencia
    dx_g, dy_g = alinear_canales(B, G)
    dx_r, dy_r = alinear_canales(B, R)
    print(f"Desplazamiento canal Verde (G): dx = {dx_g}, dy = {dy_g}")
    print(f"Desplazamiento canal Rojo (R): dx = {dx_r}, dy = {dy_r}")
    # Creamos las matrices de traslación afín
    M_g = np.float32([[1, 0, dx_g], [0, 1, dy_g]])
    M_r = np.float32([[1, 0, dx_r], [0, 1, dy_r]])

    # Aplicamos la traslación a los canales G y R
    G_alineado = cv2.warpAffine(G, M_g, (G.shape[1], G.shape[0]))
    R_alineado = cv2.warpAffine(R, M_r, (R.shape[1], R.shape[0]))

    # 3.3. Combinamos en una imagen a color (BGR por defecto)
    imagen_color_base = cv2.merge([B, G_alineado, R_alineado])

    t1 = time.time() # Tiempo final
    t_total = t1 - t0

    print(f"Tiempo final (recorte y registrado): {t_total:.4f}s")

    nombre_base = os.path.splitext(ruta_imagen)[0]

    # Guardamos versión de la imagen antes de aplicar mejoras
    ruta_salida_base = f"{nombre_base}_color_base.jpg"
    cv2.imwrite(ruta_salida_base, imagen_color_base)
    print(f"Resultado base guardado como: {ruta_salida_base}")

    # Guardamos imagen tras aplicar mejoras
    imagen_color = mejoras_extra(imagen_color_base)
    ruta_salida = f"{nombre_base}color.jpg"
    cv2.imwrite(ruta_salida, imagen_color)
    print(f"Resultado guardado como: {ruta_salida}")

    # Mostrar la imagen en pantalla (redimensionada)
    alto_disp, ancho_disp = imagen_color.shape[:2]
    escala = 800 / alto_disp # Redimension
    dim_disp = (int(ancho_disp * escala), 800)
    imagen_disp = cv2.resize(imagen_color, dim_disp, interpolation=cv2.INTER_AREA)
    
    cv2.imshow(f"Resultado - {ruta_imagen}", imagen_disp)
    cv2.waitKey(0) # Presiona cualquier tecla para cerrar la ventana
    cv2.destroyAllWindows()

# Llamada a funciones.
if __name__ == "__main__":
    lista_imagenes = ['255.jpg', '328.jpg', '499.jpg', '1099.jpg', '1246.jpg', '1822.jpg']
    for img_path in lista_imagenes:
        if os.path.exists(img_path):
            procesar_imagen(img_path)
        else:
            print(f"No se encontró la imagen: {img_path}")
