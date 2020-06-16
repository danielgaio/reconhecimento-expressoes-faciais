import numpy as np
import pandas as pd

def init_weight_and_bias(M1, M2): # M1 = input size, M2 = output size
    # os parametros de randn() são as dimenções, então vai retornar uma matriz tamanho M1 x M2 
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2) # matriz M1 x M2 com valores aleatórios dentro dividido pela raiz quadrada da soma de M1 e M2 (desvio padrão)
    b = np.zeros(M2) # viés é inicializado com zeros com tamanho M2
    return W.astype(np.float32), b.astype(np.float32) # ao retornar. converte para o tipo float32

def init_filter (shape, poolsz): # usado pelas redes neurais convolucionais
    # prod() retorna o produto de um vetor ao longo de um dado eixo. Com a dinenção especificada removida
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32) # converte pra ficar compativel com tensorflow e keras

# funções de ativação usadas dentro da rede neural
def relu(x):
    return x * (x > 0) # retorna x multiplicado por 1 ou por 0, dependendo de x ser positivo não nulo, ou não

def sigmoid(A):
    return 1 / (1 + np.exp(-A)) # 1 sobre 1 mais a exponêncial de menos a entrada

def softmax(A):
    expA = np.exp(A) # calcula a exponêncial de cada elemento de A
    return expA / expA.sum(axis=1, keepdims=True)

# Calculo da entropia crizada
def sigmoid_cost(T, Y):
    return -( T * np.log(Y) + (1-T) * np.log(1-Y) ).sum()

# versão mais comum de estropia cruzada
def cost(T, Y):
    return -(T * np.log(Y)).sum() # sum() soma todos os elementos de uma tupla e retorna um número

# versão simples um pouco mais elegante
def cost2(T, Y):
    # just uses targets to index Y instead of multiplying by a large indicator matrix with mostly zeros
    N = len(T)
    return -np.log( Y[np.arange(N), T]).sum()

def error_rate(targets, predictions): # taxa de erros entre alvos e predições
    return np.mean(targets != predictions)

# matriz de indicação com valores 0 e 1 de tamanho N x K
def y2indicator(y):
    N = len(y)
    K = len(set(y)) # retorno de set(): a set constructed from the given iterable parameter
    ind = np.zeros((N, K))

    for i in range(N):
        ind[i, y[i]] = 1
        
    return ind

def getData(balance_ones=True):
    # imagens são 48x48 = 2304 vetores de tamanho
    # N = 35887
    Y = []
    X = []
    first = True # pular primeira linha porque contém apenas headers

    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0])) # primeira coluna é o label
            X.append([int(p) for p in row[1].split()]) # segunda coluna são space separeted pixels. São transformados em inteiros
    
    X, Y = np.array(X) / 255.0, np.array(Y) # cria dois arrays numpy e normaliza

    if balance_ones:
        # balanceamento da classe 1
        X0, Y0 = X[Y!=1, :], Y[Y!=1] # separa todos os dados que não estão na classe 1
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0) # repeat(a, repeats, axis=None) -> None. Repeat elements of an array. 'a' é um array
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1))) # Join a sequence of arrays along an existing axis

    return X, Y

# recria a imagem
def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D)) # Convert a number or string to an integer, or return 0 if no arguments are given
    X = X.reshape(N, 1, d, d) # N exemplos, 1 canal de cor, largura e altura d
    return X, Y

def getBinaryData():
    Y = []
    X = []
    first = True

    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            # adiciona apenas exemplos que são das classes 0 ou 1
            if y == 0 or y ==1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    
    return np.array(X) / 255.0, np.array(Y)