import numpy as np
import matplotlib.pyplot as plt

from util import getData

# lista das possiveis categorias que a rede vai classificar as expressões
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X, Y = getData(balance_ones=False) # carrega os dados

    while True:
        for i in range(7): # loop para cada uma das 7 emoções
            x, y = X[Y==i], Y[Y==i]
            N = len(y) # pega o tamanho de y
            j = np.random.choice(N) # escolhe um valor dentro do tamanho de y
            plt.imshow(x[j].reshape(48, 48), cmap='gray') # exibe a imagem escolhida anteriormente, em P e B
            plt.title(label_map[y[j]]) # titulo da foto, o label que é uma das emoções
            plt.show() # exibe a imagem
        
        prompt = raw_input('Fechar? Pressione Y:\n') # para parar o loop
        if prompt == 'Y':
            break

if __name__ == '__main__':
    main()