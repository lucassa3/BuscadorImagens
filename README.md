# BuscadorImagens
Projeto de uma engine de busca de imagens

Relatório: Buscador de Imagens

Por Lucas Scarlato Astur

## Sobre:
O projeto foi divido em duas partes, sem conexao entre si: na primeira o objetivo foi criar uma engine de busca de imagens que recebe uma imagem e retorna as imagens mais parecidas de um banco de dados interno. A maneira como o algoritmo funciona e como se dá o pipeline da busca de uma imagem são apresentados nos tópicos relativos a parte 1. A parte 2 teve como objetivo retornar imagens mais provaveis de acordo com um termo de busca (parcial ou nao). O banco de dados utilizado foi montado pelos proprios alunos em sala de aula.

# Parte 1:

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

# Parte 2:

## Como rodar:
O usuário precisa ter instalado:
 * Python v3.6.4 ou maior;
 * Keras;
 * Tensorflow;
 * Argparse;

Para rodar o programa, simplesmente:
```
$ python search.py -s <busca>
```

Existem algumas flags opcionais para execucao do programa. Se desejar reconstruir ou construir os indices armazenados da engine de busca:

```
$ python search.py -s <busca> --build 1
```
Ou:

```
$ python search.py -s <busca> -b 1
```
Caso deseje procurar por um termo parcial de busca, ao inves da classe inteira:
```
$ python search.py -s <busca> --partial 1
```
Ou:

```
$ python search.py -s <busca> -p 1
```
O algoritmo devera retornar as 5 imagens mais confiantes pelo termo buscado, com informação de acuracia com o termo buscado.

## Funcionamento do buscador:
Esta entrega foi baseada na utilização de redes neurais convolucionais para aprender a prever classes para diferentes tipos de imagens. Em particular, nesteprojeto, foi utilizado uma rede pré treinada famosa, chamada MobileNetV2, extraido da biblioteca Keras, que é particularmente enxuta e eficiente se comparado com outras redes pré treinadas, como as redes VGG16 e VGG19. A rede utilizada foi pre treinada com o banco de dados ImageNet, que dispõe de cerca de um milhão de imagens categorizadas em 1000 classes diferentes, que são as classes aceitas pela engine de busca.

O código, em sua primeira execução, constrói os indices pro banco de dados utilizado (neste caso, um proprio montado pelos alunos da disciplina) probabilidade de cada uma das 1000 classes por imagem. Isto é feito para que estas probabilidades não tenham que ser calculadas toda vez que o algoritmo roda, o que deixaria a busca muito mais lenta. Em seguida, tambem na primeira execucao, é montado uma lista que relaciona descrição da classe com id da mesma, para evitar ficar procurando em um arquivo separado toda vez que uma busca é requisitada.

Uma vez montados os indices e a lista id-descricao, ao se buscar por um termo, este é convertido em um indice (id) que é usado para achar a probabilidade de cada imagem naquela classe buscada especifica. Por fim, é elencado as 5 imagens com a maior probabilidade para determinado indice e mostrado em janelas separadas por meio de um plt.imshow(), e caso a busca parcial apareça em mais de uma classe, todas estas classes tambem sao levadas em consideração para o resultado. O score para cada uma dessas imagens exibidas tambem é mostrado no terminal.

## Resultados:

A base de dados propria era fundamentalmente diferente da utilzada para treinamento da mobilenetv2 (iamgenet) e, portanto, poucas das imagens do dataset realmente se encaixavam no contexto das 1000 classes do imagenet. No entanto, as que se encaixavam (exemplo: banana) retornavam resultados muito bons e acuratos, com imagens chegando a quase 99% de confiança. Outras, como "hook, claw" retornou coisas como âncora o que, apesar de impreciso, faz sentido dado o formato similar. Outras buscas por classes que nao continham imagens sequer similares retornou resultados aparentemente aleatorios, mas tambem com confiança muito baixa.
