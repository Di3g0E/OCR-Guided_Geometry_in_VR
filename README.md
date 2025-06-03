# OCR-Guided Geometry in VR

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

<p align="center">
<img src="https://github.com/user-attachments/assets/6d2ae9c9-c5a7-484c-bfe8-0b79099a7b60" width="400" />
</p>

---


## 🚀 Uso
- Abre y ejecuta el notebook:
```jupyter notebook pruebas.ipynb```

- Dentro del notebook:
  - Se cargan imágenes de entrada.
  - Se aplica el OCR al texto presente en las imágenes.
  - Se determina qué figura debe dibujarse según el texto leído.
  - Se calcula la homografía para adaptar la figura a la perspectiva de la imagen.
  - La figura geométrica es dibujada directamente sobre la imagen.


---


## 🧪 Ejemplo de Uso

- Calcula el eje de la plantilla en la imagen
- Destaca los límites de la plantilla de color rojo
- Recorta la zona de la imagen donde se encuentra el texto
- OCR detecta el texto "Cubo" en la imagen
- Usando la distancia de Levenshtein minimizamos errores en las predicciones de las figuras
- Se proyecta un cubo sobre la superficie reconocida en perspectiva

- Resultado: se guarda y/o muestra la imagen con la figura dibujada

---

## 👤 Autor
Este proyecto fue desarrollado por @Di3g0E estudiante del Grado en Ingeniería de Inteligencia Artificial, como parte de la asignatura Visión Artificial (2025).
