# **How to create custom models to detect objects with YOLOv4**

This tutorial was created to show how to use YOLOv4 and darknet to create custom models to detect objects. In this repository the dataset used can be found here, the notebook can be also found here, but the notebook in this repository is a copy from a colab notebook. The colab notebook can be found here: https://colab.research.google.com/drive/14SEIafmH9Y1lksShz_wmf2fWpzbA0esT?usp=sharing.

--

Esse tutorial foi criado com objetivo de mostrar como utilizar o YOLOv4 e o darknet para criar modelos personalizados de detecção de objetos. Nesse repositorio é possivel encontrar o datast utilizado como também o notebook com todo o código, note que o notebook utilizado para esse tutorial é uma copia de um colab notebook. O colab notebook pode ser encontrado aqui: https://colab.research.google.com/drive/14SEIafmH9Y1lksShz_wmf2fWpzbA0esT?usp=sharing.

## What is YOLOv4?

O yolo foi inicialmente criado e desenvolvido por Joseph Redmon e Ali Farhadi, entretanto devido a algumas questões (Os autores originais viram que o yolo estavado sendo utilizado para fins militares) eles deixaram esse projeto de lado. Por causa disso as versões mais recentes (Yolov4 e o Yolov5) foram desenvolvidas por pessoas diferentes.

De forma resumida podemos dizer que o Yolo cria um grid na imagem (o tamanho dos quadrados do grid podem ser alterados), e após criar esse grid, o algoritmo irá chegar em cada um dos quadrados a probabilidade de ter uma classe (pessoa, cachorro...) alí, e assim colcoar uma boundingbox. Note que em mais de quadrado pode ter a mesma classe (o corpo de uma pessoa pode ocupar mais de um quadrado), então para isso foi feito uma IOU (intersection of union), ou seja, as boundingboxes que forem sendo criadas uma em cima da outra serão unificadas. E claro, como as boxes serão unificadas, a probabilidade da classe para cada uma das predições também serão somadas e depois feito a média.

fonte: https://pjreddie.com/media/files/papers/YOLOv3.pdf

# **Detecting and recognizing objects with YOLOv4 using COCO dataset**

Primeiro vamos usar o Yolov4 para prever classes do COCO usando pesos já treinado para entender melhor como funciona o Yolov4 e os passos para utiliza-lo, em seguida será feito um passo a passo de como fazer um treinamento com classes personalizadas.
