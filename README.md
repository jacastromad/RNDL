# Charla de Introducción a Redes Neuronales y Deep Learning

Este repositorio contiene los recursos utilizados en la presentación de introducción a las redes neuronales y deep learning, realizada en el IES María de Zayas y Sotomayor el 18 de marzo de 2024.

## Contenido

1. [Descripción](#descripción)
2. [Ejemplos de Código](#ejemplos-de-código)
3. [Cómo Usar](#cómo-usar)
4. [Licencia](#licencia)

## Descripción

La presentación completa está disponible en formato PDF en el archivo:

[Redes Neuronales y Deep Learning.pdf](Redes%20Neuronales%20y%20Deep%20Learning.pdf).

## Ejemplos de Código

El repositorio contiene los siguientes ejemplos de código:

- `code/mnist/`: Ejemplo Keras para clasificación de dígitos escritos a mano utilizando la base de datos MNIST.
- `code/nlp/`: Ejemplo Pytorch de procesamiento de lenguaje natural (NLP). Uso del modelo preentrenado Phi 2.

## Cómo Usar

Los ejemplos se ejecutan en un contenedor docker. Para crear la imagen puede ejecutarse el script rebuild_image.sh:
```bash
./rebuild_image.sh
```
Para ejecutar cada ejemplo, una vez creada la imagen, puede utilizarse el script run.sh:

```bash
./run.sh main.py
```

## Licencia

- La presentación en PDF está licenciada bajo la [Licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Puedes utilizar, compartir y adaptar libremente esta presentación para cualquier propósito, incluso con fines comerciales, siempre y cuando me atribuyas como autor original.

- Los ejemplos de código están dedicados al dominio público bajo la [Licencia MIT](https://opensource.org/licenses/MIT). Esto significa que puedes utilizar, modificar, distribuir y sublicenciar los ejemplos de código sin restricciones, incluso con fines comerciales, sin necesidad de atribución.


