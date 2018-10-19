# BuscadorImagens - Parte 1
Projeto de uma engine de busca de imagens

Relatório: Buscador de Imagens – Parte 1

Por Lucas Scarlato Astur

## Sobre:

O objetivo da primeira parte do projeto Buscador de Imagens é criar uma engine de busca de imagens que recebe uma imagem e retorna as imagens mais parecidas de um banco de dados interno. A maneira como o algoritmo funciona e como se dá o pipeline da busca de uma imagem são apresentados nos tópicos a seguir.


## Como rodar:
O usuário precisa ter instalado:
 * Python v3.6.4 ou maior;
 * Biblioteca opencv contrib;
```
$ pip install opencv-contrib-python==3.4.0.12
```

Para rodar o programa o usuario primeiro precisa treinar a engine com imagem no diretorio vocab (necessita descompactar o diretorio). Para isso, basta rodar:
```
$ python train_search.py
```

Em seguida, para buscar por imagens similares basta rodar:
```
$ python search.py <caminho_da_imagem>
```

O algoritmo devera retornar as 5 imagens mais parecidas do diretorio vocab, com informação de medida de distancia da imagem buscada.

## Funcionamento do buscador:
Para se buscar por uma imagem similar é necessário que um algoritmo aprenda e quantifique determinadas características tanto da imagem buscada quanto das imagens em seu banco de dados, para que possa compara-las ao escolher as mais parecidas. Para isso, a abordagem utilizada neste algoritmo é a chamada Bag of Visual Words, que recebe um vetor que descreve caracteristicas de certos pontos de uma imagem e busca separar estes pontos em um número pré-determinado de classes, através de clusterização.

![Alt text](utils/hist.png?raw=true "Title")
 
*Fig. 1 – Representação de 3 imagens (bloco acima), classes possíveis para os pontos das imagens (bloco no meio) e a frequência destes descritores em cada imagem (histogramas embaixo).*

Uma vez obtido o histograma de duas imagens, fica fácil comparar quão similar estas imagens são observando suas frequências para cada classe. A métrica de distância de histogramas utilizada é a chi-squared, representado pela seguinte fórmula:

![Alt text](utils/math.png?raw=true "Title")

Aonde xi, yi representam a frequência de cada classe i para seus respectivos histogramas.
Ao final, o objetivo é escolher as imagens mais parecidas com uma imagem de busca utilizado-se dos algoritmos descritos acima.
