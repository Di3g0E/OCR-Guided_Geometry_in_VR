# @brief main_text_ocr
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025
#

import os
import pickle
import sklearn
import argparse
from functions import OCRTrainingDataLoader, model_data, predecir_palabra, evaluate_ocr
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--train_ocr_path', default="../Materiales_Practica2/train_ocr", help='Select the training data dir for OCR')
    parser.add_argument(
        '--test_ocr_char_path', default="../Materiales_Practica2/test_ocr_char", help='Imágenes de test para OCR de caracteres')
    parser.add_argument(
        '--test_ocr_words_path', default="../Materiales_Practica2/test_ocr_words_plain", help='Imágenes de test para OCR con palabras completas')
    parser.add_argument(
        '--true_words_path', default="../Materiales_Practica2/test_ocr_words_plain/gt.txt", help='Imágenes de test para OCR con palabras completas')
    args = parser.parse_args()

    TEST_OCR_CLASSIFIER_IN_CHARS=True
    TEST_OCR_CLASSIFIER_IN_WORDS=True
    SAVED_OCR_CLF = "models/model_RandomForest.pkl"
    
    # Create the classifier reading the training data
    print("Training OCR classifier ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    #Línea original que da error -> data_ocr = template_det.data_loaders.OCRTrainingDataLoader()
    data_ocr = OCRTrainingDataLoader()
    #Línea original que da error -> if not os.path.exists(SAVED_TEXT_READER_FILE):
    if not os.path.exists(SAVED_OCR_CLF):
        # Create the directory to save the model
        os.makedirs("../Materiales_Practica2/models", exist_ok=True)

        # Load OCR training data (individual char images)
        print("Loading train char OCR data ...")
        train_data = data_ocr.load(args.train_ocr_path)
        X_train, y_train = model_data(train_data)

        # Train the OCR classifier for individual chars
        # clf = .... # POR HACER
        clf = model.fit(X_train, y_train)
        
        with open(SAVED_OCR_CLF, "wb") as pickle_file:
            pickle.dump(clf, pickle_file)

    else:
        with open(SAVED_OCR_CLF, "rb") as pickle_file:
            clf = pickle.load(pickle_file)

    if TEST_OCR_CLASSIFIER_IN_CHARS:
        # Load OCR testing data (individual char images) in args.test_char_ocr_path
        print("Loading test char OCR data ...")
        # gt_test = # POR HACER
        char_test_data = data_ocr.load(args.test_ocr_char_path, False)
        X_char_test, y_char_test = model_data(char_test_data)
        

        print("Executing classifier in char images ...")
        # estimated_test = # POR HACER
        estimated_test = clf.predict(X_char_test)
        
        # Display of classifier results statistics
        accuracy = sklearn.metrics.accuracy_score(y_char_test, estimated_test)
        print("    Accuracy char OCR = ", accuracy)

    if TEST_OCR_CLASSIFIER_IN_WORDS:
        # Load full words images for testing the words reader.
        print("Loading and processing word images OCR data ...")

        # Open results file
        results_save_path = "results_ocr_words_plain"
        try:
            os.mkdir(results_save_path)
        except:
            print('Can not create dir "' + results_save_path + '"')

        results_file = open(os.path.join(results_save_path, "results_text_lines.txt"), "w")
        
        # Execute the OCR over every single image in args.test_words_ocr_path
        # POR HACER ...

        predecir_palabra(clf, args.test_ocr_words_path, results_file, verbose=False)
   
        # Evaluar el modelo
        accuracy_lowerForm = evaluate_ocr(args.test_ocr_words_path, args.true_words_path, clf, lower_form=True, verbose=False, show_resluts=True)
        accuracy = evaluate_ocr(args.test_ocr_words_path, args.true_words_path, clf, lower_form=False, verbose=False, show_resluts=True)

        print("    Accuracy word OCR = ", accuracy)
        print("    Accuracy word OCR comparando las predicciones en minúscula = ", accuracy_lowerForm)



