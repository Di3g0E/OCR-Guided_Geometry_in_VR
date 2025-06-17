import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from functions import word_predict
from evaluar_resultados_test_ocr import levenshtein_distance

class Model3D:
    def __init__( self ):
        self.vertices = None
        self.faces = None
        self.face_centroids = None
        self.color_map = None

    def load_from_obj(self, file_path):
        """
        Adapted from: https://medium.com/@harunijaz/the-code-above-is-a-python-function-that-reads-and-loads-data-from-a-obj-e6f6e5c3dfb9

        :param file_path: full path of the .obj file to load
        """
        try:
            vertices = []
            faces = []
            with open(file_path) as f:
               for line in f:
                   if line[0] == "v":
                       vertex = list(map(float, line[2:].strip().split()))
                       vertices.append(vertex)
                   elif line[0] == "f":
                       face = list(map(int, line[2:].strip().split()))
                       faces.append(face)

            self.vertices = np.array(vertices)
            self.faces = np.array(faces)

            # Compute triangle centroids (for z-buffer plot)
            self.face_centroids = []
            for i in range(self.faces.shape[0]):
                p0 = self.faces[i, 0] - 1
                p1 = self.faces[i, 1] - 1
                p2 = self.faces[i, 2] - 1
                vertices = np.vstack((self.vertices[p0, :],
                                      self.vertices[p1, :],
                                      self.vertices[p2, :]))
                self.face_centroids.append(np.mean(vertices, axis=0))
            self.face_centroids = np.array(self.face_centroids)

        except FileNotFoundError:
            print(f"{file_path} not found.")
        except:
            print("An error occurred while loading the shape.")

    def translate(self, t):
        assert t.shape == (1, 3)

        if not self.vertices is None:
            self.vertices += t
            self.face_centroids += t

    def scale(self, scale):
        assert isinstance(scale, float)

        if not self.vertices is None:
            self.vertices *= scale
            self.face_centroids *= scale
   
    def rotate(self, angle_degrees, axis="z"):
        """
        Rota el modelo 3D en torno a un eje dado ('x', 'y', 'z') en grados.

        :param angle_degrees: Ángulo de rotación en grados.
        :param axis: Eje de rotación ('x', 'y' o 'z').
        """
        angle_radians = np.radians(angle_degrees)  # Convertir grados a radianes
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)

        # Definir la matriz de rotación según el eje elegido
        if axis == "x":
            R = np.array([[1, 0, 0],
                        [0, cos_theta, -sin_theta],
                        [0, sin_theta, cos_theta]])
        elif axis == "y":
            R = np.array([[cos_theta, 0, sin_theta],
                        [0, 1, 0],
                        [-sin_theta, 0, cos_theta]])
        else:  # Rotación en el eje Z (por defecto)
            R = np.array([[cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]])

        # Aplicar la rotación a los vértices y los centroides de las caras
        if self.vertices is not None:
            self.vertices = np.dot(self.vertices, R.T)
            self.face_centroids = np.dot(self.face_centroids, R.T)

    def plot_on_image(self, img, P):
        # Verificar si color_map existe y tiene el tamaño correcto
        if self.color_map is None or self.color_map.shape[0] < self.faces.shape[0]:
            self.color_map = np.int32(255 * np.random.rand(self.faces.shape[0], 3))  # Asegurar tamaño correcto

        # Transformación de vértices
        vertices3D = self.vertices.copy()
        vertices3D = np.hstack((vertices3D, np.ones((vertices3D.shape[0], 1))))
        vertices2D = P @ vertices3D.T
        vertices2D /= vertices2D[2, :]
        vertices2D = np.int32(np.round(vertices2D))

        # Calcular la distancia de cada triángulo a la cámara
        distances = []
        centroids = np.hstack((self.face_centroids, np.ones((self.face_centroids.shape[0], 1)))).T
        for i in range(self.faces.shape[0]):
            triangle_depth = P[2, :] @ centroids[:, i]
            triangle_depth /= np.linalg.norm(P[2, 0:3])
            triangle_depth *= np.sign(np.linalg.det(P[0:3, 0:3]))
            distances.append(triangle_depth)

        # Ordenar por distancia (z-buffer)
        distances = np.array(distances)
        dist_sorted_indices = distances.argsort()[::-1]

        # Dibujar cada triángulo en la imagen
        for i in dist_sorted_indices:
            # Asegurar que `i` no exceda los límites de `color_map`
            i = np.clip(i, 0, self.color_map.shape[0] - 1)

            p0 = self.faces[i, 0] - 1
            p1 = self.faces[i, 1] - 1
            p2 = self.faces[i, 2] - 1
            pt0 = vertices2D[0:2, p0]
            pt1 = vertices2D[0:2, p1]
            pt2 = vertices2D[0:2, p2]

            # Asignación segura de color
            color = (int(self.color_map[i, 0]), 
                    int(self.color_map[i, 1]), 
                    int(self.color_map[i, 2]))

            pts = np.vstack((pt0, pt1, pt2))
            cv2.fillPoly(img, [pts], color)

# Obtiene los KeyPoints de la imagen
def get_key_points(template_img, query_img, detector, matcher):
    keypoints_template, descriptors_template = detector.detectAndCompute(template_img, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(query_img, None)
    good_matches = get_matches(descriptors_template, descriptors_scene, matcher)

    return keypoints_template, keypoints_scene, good_matches

def get_matches(descriptors_template, descriptors_scene, matcher):
    matches = matcher.knnMatch(descriptors_template, descriptors_scene, 2)
    ratio_thresh = 0.75
    good_matches = []

    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


# Obtener la homografía del template a la escena
def get_homography_tmpl_scene(keypoints_template, keypoints_scene, good_matches):
    template = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Obtener los keypoints de los matches buenos
        template[i,0] = keypoints_template[good_matches[i].queryIdx].pt[0]
        template[i,1] = keypoints_template[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    H_t_img, _ =  cv2.findHomography(template, scene, cv2.RANSAC)   # Homografía template-scena
    return H_t_img

# Obtener la homografía de la imagen en mm (wold) al template
def get_homography_wToTemplate(template_img):
    y_img = template_img.shape[0]-1
    x_img = template_img.shape[1]-1

    pts_mm = np.float32([[0, 0], [0, 210], [185, 210], [185, 0]])    # Cogemos las distancias estableciendo x como el alto e y como el ancho por la representación de cv2
    pts_tmpl = np.float32([[0, 0], [x_img, 0], [x_img, y_img], [0, y_img]])  # puntos de las esquinas de la imagen template

    H_w_t = cv2.getPerspectiveTransform(pts_mm, pts_tmpl)

    return H_w_t

# Calcular la matriz de proyección
def compute_projection_matrix(H, K):
    """
    Calcula la matriz de proyección P = K [R | t] (3x4)
    a partir de la homografía H (3x3) y la intrínseca K (3x3).
    """
    # 1. Inversa de K por H
    K_inv = np.linalg.inv(K)
    H_norm = K_inv @ H

    # 2. Escalado
    h1, h2, h3 = H_norm[:, 0], H_norm[:, 1], H_norm[:, 2]
    lambda_ = 1.0 / np.linalg.norm(h1)
    r1 = lambda_ * h1
    r2 = lambda_ * h2
    t  = lambda_ * h3

    # 3. Ortonormalización de R
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))   

    # 4. Formar P
    Rt = np.column_stack((R, t))
    P = K @ Rt

    return P

# Imagen query mostrada con las dimensiones reales
def img_to_world(query_img, H_img_w, dim=(210, 185), roi_start=210, roi_end=292):
    # Aplicar la transformación de perspectiva
    transformed_img = cv2.warpPerspective(query_img, H_img_w, dim)

    # Recortar la imagen en el rango deseado
    roi_img = transformed_img[roi_start:roi_end, :]  # Recorta filas de 220 a 290
    img = cv2.flip(roi_img, 0)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# Mostrar ejes origen
def view_orig(img, P, axes_len):
    # 1. Definimos los 4 puntos homogéneos (origen, X, Y, Z)
    pts_orig_hom = np.array([
        [0,          0,          0, 1],    # Origen
        [axes_len,   0,          0, 1],    # Eje X
        [0,   axes_len,          0, 1],    # Eje Y
        [0,          0,   axes_len, 1]     # Eje Z
    ])

    # 2. Proyectar a 2D con la homografía
    #    pts_proj será 3×4, cada columna [x, y, w]
    pts_proj = P @ pts_orig_hom.T
    pts_proj /= pts_proj[2, :]            # normalizamos por w

    # 3. Calculamos los vectores de cada eje en la imagen
    origin2d = pts_proj[:2, 0]            # (x,y) del origen
    vec_X    = pts_proj[:2, 1] - origin2d # dirección X
    vec_Y    = pts_proj[:2, 2] - origin2d # dirección Y
    vec_Z    = pts_proj[:2, 3] - origin2d # dirección Z

    # 5. Convertimos a tuplas de int para cv2.line
    origin_pt  = tuple(origin2d.astype(int))
    x_axis_end = tuple(pts_proj[:2, 1].astype(int))
    y_axis_end = tuple(pts_proj[:2, 2].astype(int))
    z_axis_end = tuple(pts_proj[:2, 3].astype(int))

    # 6. Dibujamos sobre la imagen
    cv2.line(img, origin_pt, x_axis_end, (0, 0, 255), 4)    # eje X en rojo (BGR)
    cv2.line(img, origin_pt, y_axis_end, (0, 255, 0), 4)    # eje Y en verde
    cv2.line(img, origin_pt, z_axis_end, (255, 0, 0), 4)    # eje Z en azul

    return img

# Mostrar bordes
def view_edge(img, H, pts_hom):
    # Proyectar puntos a 2D
    limit_pts = H @ pts_hom.T
    limit_pts /= limit_pts[2]     # Normalizar

    # Convertir los puntos a int para usarlos en cv2.line
    limit_pts = limit_pts[:2, :].T  # Ahora es (4,2): cada fila es (x,y)

    # Añadir el primer punto al final para cerrar el borde
    limit_pts = np.vstack([limit_pts, limit_pts[0]])

    # Dibujar las líneas conectando los puntos
    for i in range(len(limit_pts) - 1):
        pt1 = tuple(limit_pts[i].astype(int))
        pt2 = tuple(limit_pts[i + 1].astype(int))
        cv2.line(img, pt1, pt2, color=(255, 0, 255), thickness=4)  # Magenta en BGR

    return img

# Guardamos las imágenes
def save_result_image(image, images_path, input_filename):
    # Crear el directorio 'resultado_imgs' si no existe
    result_dir = os.path.join(images_path, "resultado_imgs")
    os.makedirs(result_dir, exist_ok=True)

    # Generar el nombre del archivo de salida
    output_filename = f"resultado_{os.path.basename(input_filename)}"
    output_path = os.path.join(result_dir, output_filename)

    # Guardar la imagen
    cv2.imwrite(output_path, image)

# Devuelve la figura con la distancia de Levenshtein más cercana
def figura_mas_cercana(palabra):
    figuras = ["cubo", "dodecaedro", "icosaedro", "octaedro", "tetraedro"]
    distancias = [(figura, levenshtein_distance(palabra.lower(), figura)) for figura in figuras]
    figura_cercana = min(distancias, key=lambda x: x[1])
    return figura_cercana[0]


def show_figure(img_path, figures_path, detector, matcher, model, limit_pts_hom, pts_mm_hom, fig_class, show_results=False):
    # Cargamos la imagen de la plantilla
    template_img_path = os.path.join(img_path, "template_cropped.png")
    template_img = cv2.imread(template_img_path)
    if template_img is None:
            print("No puedo encontrar la imagen " + template_img_path)

    # Cargamos la matriz de intrínsecos de la cámara
    K = np.loadtxt(os.path.join(img_path, "intrinsics.txt"))

    # Cargamos el directorio de las imágenes
    paths = sorted(glob(os.path.join(img_path, "*.jpg")))

    # Inicializamos el error
    error_acumulado = 0
    img_count = 0

    # Iteramos por cada imagen
    for f in paths:
        query_img_path = f
        if not os.path.isfile(query_img_path):
                continue

        query_img = cv2.imread(query_img_path)
        if query_img is None:
                print("No puedo encontrar la imagen " + query_img_path)
                continue

        # Localizar el plano en la imagen y calcular R, t -> P
        # 1. Calcular Key Points para calcular las homografías
        keypoints_template, keypoints_scene, good_matches = get_key_points(template_img, query_img, detector, matcher)

        # 2. Calcular las Homografías
        H_t_img = get_homography_tmpl_scene(keypoints_template, keypoints_scene, good_matches)
        H_w_t = get_homography_wToTemplate(template_img)
        H_w_img = H_t_img @ H_w_t

        # 3. Calcular la matriz de proyección
        P = compute_projection_matrix(H_w_img, K)

        # Mostrar la imagen con los bordes y el origen
        img = query_img.copy()
        img = view_edge(img, H_w_img, limit_pts_hom) # Muestra el borde exterior
        img = view_edge(img, H_w_img, pts_mm_hom)    # Muestra el borde donde debe ir el cubo

        # Mostrar los ejes del sistema de referencia de la plantilla sobre una copia de la imagen de entrada.
        img = view_orig(img, P, 20)

        # Obtener imagen para OCR
        H_img_w = np.linalg.inv(H_w_img)
        ocr_img = img_to_world(query_img, H_img_w, dim=(195, 290), roi_start=210, roi_end=292)

        word, _ = word_predict(ocr_img, model, False)  # La predicción no es muy buena pero vamos a intentar que se acerce seleccionando la figura más cercana por la distancia de Levenshtein
        figure = figura_mas_cercana(word)

        # Cargar el modelo 3D del cubo y colocarlo en el lugar pedido.
        model_3d_file = os.path.join(figures_path, f"{figure}.obj")
        print("Cargando el modelo 3D " + model_3d_file)

        fig_class.load_from_obj(file_path=model_3d_file)
        escala = 43.0/2.0
        traslacion = np.array([[92.5, 126.5, escala]]) # Traslación en XYZ
        if figure == "octaedro": 
            fig_class.rotate(angle_degrees=45, axis="z")
        fig_class.scale(scale=escala)
        fig_class.translate(t=traslacion)


        # Mostrar el modelo 3D del cubo sobre la imagen plot_img
        fig_class.plot_on_image(img=img, P=P)

        # Mostrar el resultado en pantalla.
        if show_results:
            # plt.figure(figsize=(30, 15))
            # plt.subplot(1, 3, 1)
            # plt.imshow(query_img)
            # plt.subplot(1, 3, 2)
            # plt.imshow(ocr_img)
            # plt.subplot(1, 3, 3)
            # plt.imshow(img)
            # plt.show()

            cv2.imshow("3D info on images", cv2.resize(img, None, fx=0.3, fy=0.3))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Guardar imagenes
        save_result_image(img, img_path, f"{figure}_{os.path.basename(f)}")
