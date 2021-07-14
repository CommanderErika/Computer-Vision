# **How to create custom models to detect objects with YOLOv4**

This tutorial was created to show how to use YOLOv4 and darknet to create custom models to detect objects. In this repository the dataset used can be found here, the notebook can be also found here, but the notebook in this repository is a copy from a colab notebook. The colab notebook can be found here: [Colab Notebook](https://colab.research.google.com/drive/14SEIafmH9Y1lksShz_wmf2fWpzbA0esT?usp=sharing).

--

Esse tutorial foi criado com objetivo de mostrar como utilizar o YOLOv4 e o darknet para criar modelos personalizados de detecção de objetos. Nesse repositorio é possivel encontrar o datast utilizado como também o notebook com todo o código, note que o notebook utilizado para esse tutorial é uma copia de um colab notebook. O colab notebook pode ser encontrado aqui: [Colab Notebook](https://colab.research.google.com/drive/14SEIafmH9Y1lksShz_wmf2fWpzbA0esT?usp=sharing).

## What is YOLOv4?

O yolo foi inicialmente criado e desenvolvido por Joseph Redmon e Ali Farhadi, entretanto devido a algumas questões (Os autores originais viram que o yolo estavado sendo utilizado para fins militares) eles deixaram esse projeto de lado. Por causa disso as versões mais recentes (Yolov4 e o Yolov5) foram desenvolvidas por pessoas diferentes.

De forma resumida podemos dizer que o Yolo cria um grid na imagem (o tamanho dos quadrados do grid podem ser alterados), e após criar esse grid, o algoritmo irá chegar em cada um dos quadrados a probabilidade de ter uma classe (pessoa, cachorro...) alí, e assim colcoar uma boundingbox. Note que em mais de quadrado pode ter a mesma classe (o corpo de uma pessoa pode ocupar mais de um quadrado), então para isso foi feito uma IOU (intersection of union), ou seja, as boundingboxes que forem sendo criadas uma em cima da outra serão unificadas. E claro, como as boxes serão unificadas, a probabilidade da classe para cada uma das predições também serão somadas e depois feito a média.

fonte: https://pjreddie.com/media/files/papers/YOLOv3.pdf

# **Detecting and recognizing objects with YOLOv4 using COCO dataset**

First to understand better how YOLOv4 works we'll be using YOLOv4 to predict classes from COCO dataset, and we'll be using pre-trained weights.
All the code used can be found in the colab notebook mentioned before.

--

Primeiro vamos usar o Yolov4 para prever classes do COCO usando pesos já treinado para entender melhor como funciona o Yolov4 e os passos para utiliza-lo, em seguida será feito um passo a passo de como fazer um treinamento com classes personalizadas.
Todo código utilizado pode ser encontra no colab notebook mencionado anteriormente.

## Downloading YOLOv4 files

Agora iremos fazer o download do Yolov4 direto do repositorio. Em seguida será feito um %cd para o arquivo que acabamos de baixar o darknet, é nesse arquivo onde está todos os arquivos do yolov4, e logo em seguida iremos alterar o arquivo makefile para que yolov4 opere usando o opencv e a GPU.

* É possivel fazer o treinamento e a detecção do Yolov4 usando apenas a CPU, entretanto é bastante demorado e se recomendo utilizar a GPU.
*  O OpenCV é utilizado para gerar os boundingBoxes durante a detecção numa imagem ou video.

`# Git clone no repositorio`

> `git clone https://github.com/AlexeyAB/darknet`

`# Alterando o makefile para habilitar a GPU e o OpenCV`

> `cd darknet`
> 
> `sed -i 's/OPENCV=0/OPENCV=1/' Makefile`
> 
> `sed -i 's/GPU=0/GPU=1/' Makefile`
> 
> `sed -i 's/CUDNN=0/CUDNN=1/' Makefile`
> 
> `sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile`

## Download pre-treined files and makefile

> `wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights`

> `make`

## Detecting and Recognizing Objects 

> `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg`

After doing all this steps we can detect any class with COCO dataset

# **Creating custom model with YOLOv4**

Agora iremos fazer o passo a passo de como fazer um treinamento com classes personalizadas. Para isso é necessário fazer a criação/alteração de alguns arquivos importantes para o nosso treinamento, eles são:

* Dados customizados com suas respectivas labels (as bounding boxes).
* Arquivo .cfg onde irá conter a estrutura do modelo de treinamento.
* Arquivos .data e .names, onde o .data irá conter as informações sobre os diretorios das imagens (tanto de validação como de treinamento), os pesos, e até mesmo o diretorio do nosso arquivo .names, e o arquivo .names irá conter as classes existentes para esse treinamento.
* Arquivos train.txt e o test.txt, que são os arquivos que indicam quais são os nomes das nossas imagens e os respectivos diretorios. Esses arquivos também são informados no arquivo .data.

Para criar um modelo personalizados iremos baixar os arquivo do YOLOv4 como fizemos em etapas anteriores e faremos o make do arquivo, e em seguida iremos baixar um peso pre-treinado feito para fazer treinamento de modelos.

## Download pre-treined weights to train the model

`# Baixando pesos pre-treinados do Yolo`
> `wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137`

## Dataset and Labels

Agora que fizemos alguns pequenos preparativos vamos começar a trabalhar em cima do nosso dataset. Para esse projeto foi utilziado o dataset do MIT Indoor, como há varias imagens referentes a lugares fechados, iremos utilziar apenas as imagens referentes à banheiros, em especial serão criadas três classes: bath (banheira), sink (pia), e toilet (vazo sanitário), e para cada uma dessas classes será criada uma label na imagem.

Dentro da pasta darknet será criado uma pasta chamada dataset, onde irá conter todos os nosso arquivos e as imagens necessárias para o nosso treinamento. Na figura a seguir motra como fica a arquitetura dos arquivos para melhor entendimentos:

![1](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/1.png?raw=true)

**O dataset pode ser encontrado nesse link: http://web.mit.edu/torralba/www/indoor.html**

`# Criando o nosso diretório dataset`

> `mkdir dataset`

Agora que criamos o nosso diretório dataset, será criado dois diretórios. Esses dois novos diretórios são para organizar as imagens e as labels, sendo que um diretório que será para as imagens de treino e outro diretório será para as imagens de validação. Ao criar esses dois diretórios, coloque as imagens dentro de cada um dos diretórios.

Para esse projeto será utilizado cerca de 100 imagens, então 90% das imagens serão para treino e os outros 10% para validação.

* Total de imagens - 100 imagens.
* Treino - 90 imagens.
* Validação - 10 imagens.

`mkdir dataset/train
mkdir dataset/validation`

### Creating Labels

Para gerar as labels de cada um dos objetos na imagem foi utilizado um programa chamado labelImage. Para criar cada uma das BoudingBoxes bastar usar a opção de criar 'RectBox' do labelImg e em seguida colocar cada tag da classe. Ao fazer isso para cada uma as classes existentem na imagem basta salvar todo o processo no botão 'Save', lembrando que antes de salvar devemos por o arquivo no tipo 'Yolo'.

![LabelImg](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/4.png?raw=true)

Note que quando salvarmos o arquivo, será erado um arquivo de tipo .txt com o mesmo nome que a imagem.

**É possivel encontrar e baixar o labelImg no link: https://github.com/tzutalin/labelImg**

## Generating train.txt and train.txt files

Esses arquivos (train.txt e o test.txt) são usados para dizer ao Yolov4 onde é o diretorio das nossas imagens e quais são os nomes das imagens. Aqui abaixo pe possivel ver um exemplo de arquivo .txt para o train e test.

![Train txt](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/2.png?raw=true)
![Test txt](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/3.png?raw=true)

`# Gerando o arquivo train.txt

import os
os.getcwd()
collection = 'dataset/teste'

file = open('dataset/teste.txt', "w+")

for filename in os.listdir(collection):
  if(filename[-3:] != 'txt'):
    file.write('dataset/teste/' + filename + '\n')

file.close()`

`# Gerando os arquivos test.txt

os.getcwd()
collection = 'dataset/validation'

file = open('dataset/test.txt', "w+")

for filename in os.listdir(collection):
  if(filename[-3:] != 'txt'):
    file.write('dataset/test/' + filename + '\n')

file.close()`

## File .data and .names

Agora que fizemos os ultimos arquivos .txt restante, vamos criar os arquivos .data e .names. O arquivo .data vai conter informaçõe sobre os diretorios para o Yolov4, enquanto o arquivo .names vai conter informaçõs sobre os nomes das classes.

**Arquivo .names**

![Names](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/5.png?raw=true)

**Arquivo .data**

![Data](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/6.png?raw=true)

## .cfg file

Por fim, o ultimo arquivo que temos que criar o .cfg. Esse tipo de arquivo contém as informações referentes a rede, e as possiveis configurações que podem ser feitas nela. Para facilitar sugiro pegar o arquivo coco.cfg na pasta cfg do darknet, e em seguida será feita umas pequenas modificações simples para se adaptar a nossas classes.

Dentro do arquivo .cfg faça as seguintes mudanças:

* max_batches = 2000 * (Número de classes)
* steps = max_batches * 0.8, max_batches * 0.9
* Em cada camada Yolo altere o valor classes = Número de classes
* Antes de cada camada Yolo, altere o filtro da camada convolucional para filters = (Número de classes + 5) * 3

## Our dataset

![Examples](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/7.png?raw=true)

## Training our model

`./darknet detector train dataset/classes.data dataset/yolov4-custom.cfg yolov4.conv.137 -dont_show -map`

## Predicting

`img_path = "dataset/room30.jpg"`

`./darknet detector test dataset/classes.data dataset/yolov4-custom.cfg dataset/yolov4-custom_best.weights {img_path} -dont-show`

![Examples](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/8.png?raw=true)

![Examples](https://github.com/CommanderErika/Computer-Vision/blob/main/How%20to%20use%20Yolov4/yolov4%20tutorial/9.png?raw=true)
