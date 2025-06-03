# OCR-Guided Geometry

Este proyecto tiene como objetivo entrenar y evaluar diferentes algoritmos de OCR (Reconocimiento Óptico de Caracteres) para identificar texto en imágenes. A partir del texto leído, se calcula la homografía necesaria para proyectar una figura geométrica sobre la imagen, correspondiente al nombre identificado.

Este trabajo se desarrolla como parte de la asignatura de **Visión Artificial**, en el grado de **Ingeniería de Inteligencia Artificial**.

---

## 🧠 Objetivo del Proyecto

- Evaluar y comparar el rendimiento de distintos métodos de OCR.
- Calcular homografías entre coordenadas de referencia y puntos reales en la imagen.
- Dibujar automáticamente la figura geométrica mencionada en el texto leído, en la imagen de entrada.

---

## 🔍 Ejemplo Visual

A continuación se muestra un ejemplo del funcionamiento del sistema, incluyendo la detección del texto, cálculo de homografía y dibujo de la figura correspondiente:

![demo](./images/demo_result.png)

---

## 🧰 Tecnologías Utilizadas

- **Python 3.9+**
- `OpenCV`
- `Tesseract OCR`
- `matplotlib`
- `numpy`
- `Pillow`

---

## ⚙️ Instalación
1. Clona el repositorio:
git clone https://github.com/tuusuario/OCR-Guided-Geometry.git
cd OCR-Guided-Geometry

2. (Opcional) Crea un entorno virtual:
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate     # En Windows

3. Instala las dependencias:
pip install -r requirements.txt
Asegúrate de tener instalado Tesseract-OCR en tu sistema. Puedes encontrarlo en: https://github.com/tesseract-ocr/tesseract


---


## 🚀 Uso
- Abre y ejecuta el notebook:
jupyter notebook pruebas.ipynb

- Dentro del notebook:
Se cargan imágenes de entrada.
Se aplica el OCR al texto presente en las imágenes.
Se determina qué figura debe dibujarse según el texto leído.
Se calcula la homografía para adaptar la figura a la perspectiva de la imagen.
La figura geométrica es dibujada directamente sobre la imagen.


---


## 🧪 Ejemplo de Uso

- OCR detecta el texto "TRIÁNGULO" en la imagen
- Se proyecta un triángulo sobre la superficie reconocida en perspectiva

- Resultado: se guarda o muestra la imagen con la figura dibujada

---

## 👤 Autor
Este proyecto fue desarrollado por @Di3g0E estudiante del Grado en Ingeniería de Inteligencia Artificial, como parte de la asignatura Visión Artificial (2025).
