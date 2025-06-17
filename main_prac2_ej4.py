import cv2
import pickle
import numpy as np
from functions_ej4 import Model3D, show_figure


if __name__ == '__main__':
    # Cargamos las direcciones de los directorios
    test_ocr_simple_path = "test_template_ocr_simple"
    test_ocr_path = "test_template_ocr"
    figures_path = "3d_models"
    model_ocr = "models/model_RandomForest.pkl"

    # Cargamos el detector y el matcher
    nfeatures = 4000
    detector = cv2.SIFT_create(nfeatures)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    # Cargamos el modelo OCR
    with open(model_ocr, "rb") as pickle_file:
        clf = pickle.load(pickle_file)

    # Inicializamos la clase Modelo3d
    m3D = Model3D()

    # Inicializamos los límites
    limit_pts_hom = np.float32([[0,   0,   1], 
                            [185, 0,   1], 
                            [185, 210, 1], 
                            [0,   210, 1]         
    ])
    pts_mm_hom = np.float32([[71,  105, 1], 
                            [71,  148, 1], 
                            [114, 148, 1], 
                            [114, 105, 1]       
    ])

    # Imágenes de Test_template_ocr_simple
    show_figure(test_ocr_simple_path, figures_path, detector, matcher, clf, limit_pts_hom, pts_mm_hom, m3D, True)
    print()
    # Imágenes de Test_template_ocr
    show_figure(test_ocr_path, figures_path, detector, matcher, clf, limit_pts_hom, pts_mm_hom, m3D, False)
