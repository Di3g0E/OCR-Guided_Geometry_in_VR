# Librerías
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score


class OCRTrainingDataLoader:

    def __init__(self, char_size=(30,30)):
        self.name = 'URJC-OCR-TRAIN'
        self.char_size = char_size

    def load(self, data_path, show_results=False):
        images = dict()
        for root, _, file_names in os.walk(data_path):
            tag = os.path.basename(root)
            print("====> Loading ", tag, " images.")
            images[tag] = self.__load_images(root, file_names, show_results)

        return images
    
    def __load_images(self, data_path, file_names, show_results=False, extensiones_validas={".jpg", ".jpeg", ".png"}):
        images = []
        for name in file_names:
            if not os.path.splitext(name)[1].lower() in extensiones_validas:    # no tiene la extensión adecuada
                print("*** ERROR: Invalid extension " + name)
                continue
            
            I = cv2.imread(os.path.join(data_path, name), 1)    # cargamos la imagen en escala de grises.
            if not type(I) is np.ndarray:   # no es una imagen.
                print("*** ERROR: Couldn't read image " + name)
                continue

            bin_img, contours = process_img(I, name, show_results)
            vector = cut_img(bin_img, max(contours, key=cv2.contourArea), self.char_size)
            images.append(vector)

        return images
    

def process_img(img, name=None, show_results=False):
    # 1. Cargamos la imagen en escala de grises
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Umbralizar la imagen
    bin_img = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Encontrar los bordes
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE

    # Visualización
    if show_results and name == '0000.png':
        # Dibujar los contornos en una copia de la imagen original
        imagen_contornos = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(imagen_contornos, contours, -1, (0, 255, 0), 2)

        # Mostrar resultados (cv2)
        cv2.imshow('Imagen Original', img)
        cv2.imshow('Imagen Binaria', bin_img)
        cv2.imshow('Contornos', imagen_contornos)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Mostrar resultados (plt)
        # _, axs = plt.subplots(1, 3, figsize=(15, 5))

        # axs[0].imshow(np.rot90(img), cmap='gray')
        # axs[0].set_title('Imagen Original')
        # axs[0].axis('off')

        # axs[1].imshow(np.rot90(bin_img), cmap='gray')
        # axs[1].set_title('Imagen Binaria')
        # axs[1].axis('off')

        # axs[2].imshow(np.rot90(imagen_contornos))
        # axs[2].set_title('Contornos Detectados')
        # axs[2].axis('off')

        # plt.show()

    return bin_img, contours

def cut_img(bin_img, contour, size=(30,30)):
    x, y, w, h = cv2.boundingRect(contour)
    cut = bin_img[y:y+h, x:x+w]
    roi = cv2.resize(cut, size)
    
    return roi.flatten()
          
def word_predict(img, model, show_results=False):
    bin_img, contours = process_img(img, show_results=True)

    # Paso 1: Obtener todas las alturas para calcular umbral
    medidas = [cv2.boundingRect(contour)[3] for contour in contours if cv2.boundingRect(contour)[2] > 5 and cv2.boundingRect(contour)[3] > 5]
    if not medidas:
        return "", (0, 0, img.shape[1], img.shape[0])  # Devuelve caja completa si no hay letras

    altura_media = sum(medidas) / len(medidas)
    umbral_altura = altura_media * 0.5  # Filtrar contornos demasiado pequeños

    # Paso 2: Filtrar y procesar contornos válidos
    letras = []
    posiciones = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5 and h >= umbral_altura:
            vector = cut_img(bin_img, contour)
            letras.append((x, vector))
            posiciones.append((x, y, w, h))

    # Ordenar de izquierda a derecha
    letras.sort(key=lambda tup: tup[0])
    posiciones.sort(key=lambda tup: tup[0])
    vectores = [v for _, v in letras]

    palabra = "".join(model.predict(vectores)) if vectores else ""

    if posiciones:
        xs = [x for x, _, w, _ in posiciones]
        ys = [y for _, y, _, _ in posiciones]
        xws = [x + w for x, _, w, _ in posiciones]
        yhs = [y + h for _, y, _, h in posiciones]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xws), max(yhs)
    else:
        x_min, y_min, x_max, y_max = 0, 0, img.shape[1], img.shape[0]  # Caja completa por defecto

    if show_results and posiciones:
        # Rectángulo rojo envolvente de la palabra
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

        # Dibujar los rectángulos amarillos en cada letra
        for x, y, w, h in posiciones:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Mostrar cada letra predicha directamente debajo de su respectiva letra original
        for (x, y, w, h), letra in zip(posiciones, palabra):
            baseline_y = y + h + 30 # Ajuste para colocar la letra debajo del rectángulo amarillo
            cv2.putText(img, letra, (x + w // 4, baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            
        cv2.imshow("Reconocimiento OCR", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return palabra, (x_min, y_min, x_max, y_max)


def model_data(data):
    X, y = [], []

    for value, img_list in data.items():
        for img in img_list or []:
            X.append(img)
            y.append(value)
            
    return np.array(X), np.array(y)

def evaluate_ocr(word_test_path, gt_path, model, lower_form=True, verbose=False, show_resluts=False):
    # Cargar las respuestas correctas desde gt.txt
    gt_dict = {}

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            filename = parts[0]  # Nombre de la imagen
            ground_truth = parts[-1]  # Última columna con la respuesta correcta
            gt_dict[filename] = ground_truth

    y_pred = []
    y_true = []

    # Procesar las imágenes y comparar predicciones
    for root, _, file_names in os.walk(word_test_path):
        for name in file_names:
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
                img_path = os.path.join(root, name)
                img = cv2.imread(img_path, 1)

                if img is None:
                    print(f"Error al cargar la imagen: {name}")
                    continue
                
                if name != '0000.png': show_resluts = False
                pred, _ = word_predict(img, model, show_resluts)
                ground_truth = gt_dict.get(name, "Desconocido")  # Obtener respuesta correcta

                if lower_form:
                    y_pred.append(pred.lower())
                    y_true.append(ground_truth.lower())
                    if verbose:
                        print(f"Imagen: {name} | Predicción: {pred.lower()} | Ground Truth: {ground_truth.lower()}")

                else:
                    y_pred.append(pred)
                    y_true.append(ground_truth)
                    if verbose:
                        print(f"Imagen: {name} | Predicción: {pred} | Ground Truth: {ground_truth}")

    # Calcular precisión
    accuracy = accuracy_score(y_true, y_pred) * 100
    return accuracy

def predecir_palabra(model, input_folder, results_file, verbose=False):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 1)

            if img is None:
                print(f"[ERROR] No se pudo leer la imagen: {img_path}")
                continue

            texto_ocr, (x1, y1, x2, y2) = word_predict(img, model, show_results=False)
            linea_resultado = f"{filename};{x1};{y1};{x2};{y2};{texto_ocr}\n"
            results_file.write(linea_resultado)
            if verbose:
                print(f"[INFO] Procesado: {filename} -> {texto_ocr}")


# Funciones y clases
# class OCRTrainingDataLoader:

#     def __init__(self, char_size=(30,30)):
#         self.name = 'URJC-OCR-TRAIN'
#         self.char_size = char_size

#     def load(self, data_path, show_results=False):
#         images = dict()
#         for root, _, file_names in os.walk(data_path):
#             tag = os.path.basename(root)
#             print("====> Loading ", tag, " images.")
#             images[tag] = self.__load_images(root, file_names, show_results)

#         return images
    
#     def __load_images(self, data_path, file_names, show_results=False, extensiones_validas={".jpg", ".jpeg", ".png"}):
#         images = []
#         for name in file_names:
#             if not os.path.splitext(name)[1].lower() in extensiones_validas:    # no tiene la extensión adecuada
#                 print("*** ERROR: Invalid extension " + name)
#                 continue
            
#             I = cv2.imread(os.path.join(data_path, name), 1)    # cargamos la imagen en escala de grises.
#             if not type(I) is np.ndarray:   # no es una imagen.
#                 print("*** ERROR: Couldn't read image " + name)
#                 continue

#             bin_img, contours = process_img(I, name, show_results)
#             vector = cut_img(bin_img, max(contours, key=cv2.contourArea), self.char_size)
#             images.append(vector)

#         return images
    

# def process_img(img, name=None, show_results=False):
#     # 1. Cargamos la imagen en escala de grises
#     grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 2. Umbralizar la imagen
#     bin_img = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#     # 3. Encontrar los bordes
#     contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE

#     # Visualización
#     if show_results and name == '0000.png':
#         # Dibujar los contornos en una copia de la imagen original
#         imagen_contornos = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         cv2.drawContours(imagen_contornos, contours, -1, (0, 255, 0), 2)

#         # Mostrar resultados (cv2)
#         cv2.imshow('Imagen Original', img)
#         cv2.imshow('Imagen Binaria', bin_img)
#         cv2.imshow('Contornos', imagen_contornos)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         # Mostrar resultados (plt)
#         # _, axs = plt.subplots(1, 3, figsize=(15, 5))

#         # axs[0].imshow(np.rot90(img), cmap='gray')
#         # axs[0].set_title('Imagen Original')
#         # axs[0].axis('off')

#         # axs[1].imshow(np.rot90(bin_img), cmap='gray')
#         # axs[1].set_title('Imagen Binaria')
#         # axs[1].axis('off')

#         # axs[2].imshow(np.rot90(imagen_contornos))
#         # axs[2].set_title('Contornos Detectados')
#         # axs[2].axis('off')

#         # plt.show()

#     return bin_img, contours

# def cut_img(bin_img, contour, size=(30,30)):
#     x, y, w, h = cv2.boundingRect(contour)
#     cut = bin_img[y:y+h, x:x+w]
#     roi = cv2.resize(cut, size)
    
#     return roi.flatten()
          
# def word_predict(img, model, show_results=False):
#     bin_img, contours = process_img(img, show_results=True)

#     # Paso 1: Obtener todas las alturas para calcular umbral
#     medidas = [cv2.boundingRect(contour)[3] for contour in contours if cv2.boundingRect(contour)[2] > 5 and cv2.boundingRect(contour)[3] > 5]
#     if not medidas:
#         return "", (0, 0, img.shape[1], img.shape[0])  # Devuelve caja completa si no hay letras

#     altura_media = sum(medidas) / len(medidas)
#     umbral_altura = altura_media * 0.5  # Filtrar contornos demasiado pequeños

#     # Paso 2: Filtrar y procesar contornos válidos
#     letras = []
#     posiciones = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 5 and h > 5 and h >= umbral_altura:
#             vector = cut_img(bin_img, contour)
#             letras.append((x, vector))
#             posiciones.append((x, y, w, h))

#     # Ordenar de izquierda a derecha
#     letras.sort(key=lambda tup: tup[0])
#     posiciones.sort(key=lambda tup: tup[0])
#     vectores = [v for _, v in letras]

#     palabra = "".join(model.predict(vectores)) if vectores else ""

#     if posiciones:
#         xs = [x for x, _, w, _ in posiciones]
#         ys = [y for _, y, _, _ in posiciones]
#         xws = [x + w for x, _, w, _ in posiciones]
#         yhs = [y + h for _, y, _, h in posiciones]
#         x_min, y_min = min(xs), min(ys)
#         x_max, y_max = max(xws), max(yhs)
#     else:
#         x_min, y_min, x_max, y_max = 0, 0, img.shape[1], img.shape[0]  # Caja completa por defecto

#     if show_results and posiciones:
#         # Rectángulo rojo envolvente de la palabra
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

#         # Dibujar los rectángulos amarillos en cada letra
#         for x, y, w, h in posiciones:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

#         # Mostrar cada letra predicha directamente debajo de su respectiva letra original
#         for (x, y, w, h), letra in zip(posiciones, palabra):
#             baseline_y = y + h + 30 # Ajuste para colocar la letra debajo del rectángulo amarillo
#             cv2.putText(img, letra, (x + w // 4, baseline_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            
#         cv2.imshow("Reconocimiento OCR", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return palabra, (x_min, y_min, x_max, y_max)


# def model_data(data):
#     X, y = [], []

#     for value, img_list in data.items():
#         for img in img_list or []:
#             X.append(img)
#             y.append(value)
            
#     return np.array(X), np.array(y)

# def evaluate_ocr(word_test_path, gt_path, model, lower_form=True, verbose=False, show_resluts=False):
#     # Cargar las respuestas correctas desde gt.txt
#     gt_dict = {}

#     with open(gt_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(";")
#             filename = parts[0]  # Nombre de la imagen
#             ground_truth = parts[-1]  # Última columna con la respuesta correcta
#             gt_dict[filename] = ground_truth

#     y_pred = []
#     y_true = []

#     # Procesar las imágenes y comparar predicciones
#     for root, _, file_names in os.walk(word_test_path):
#         for name in file_names:
#             if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
#                 img_path = os.path.join(root, name)
#                 img = cv2.imread(img_path, 1)

#                 if img is None:
#                     print(f"Error al cargar la imagen: {name}")
#                     continue
                
#                 if name != '0000.png': show_resluts = False
#                 pred, _ = word_predict(img, model, show_resluts)
#                 ground_truth = gt_dict.get(name, "Desconocido")  # Obtener respuesta correcta

#                 if lower_form:
#                     y_pred.append(pred.lower())
#                     y_true.append(ground_truth.lower())
#                     if verbose:
#                         print(f"Imagen: {name} | Predicción: {pred.lower()} | Ground Truth: {ground_truth.lower()}")

#                 else:
#                     y_pred.append(pred)
#                     y_true.append(ground_truth)
#                     if verbose:
#                         print(f"Imagen: {name} | Predicción: {pred} | Ground Truth: {ground_truth}")

#     # Calcular precisión
#     accuracy = accuracy_score(y_true, y_pred) * 100
#     return accuracy

# def predecir_palabra(model, input_folder, results_file):
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#             img_path = os.path.join(input_folder, filename)
#             img = cv2.imread(img_path, 1)

#             if img is None:
#                 print(f"[ERROR] No se pudo leer la imagen: {img_path}")
#                 continue

#             texto_ocr, (x1, y1, x2, y2) = word_predict(img, model, show_results=False)
#             linea_resultado = f"{filename};{x1};{y1};{x2};{y2};{texto_ocr}\n"
#             results_file.write(linea_resultado)
#             # print(f"[INFO] Procesado: {filename} -> {texto_ocr}")
