import numpy as np 
import matplotlib.pyplot as plt
import csv

def showPlot(n, ws, dataset_name, step):
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    i = 0
    with open(dataset_name + '/results_k=' + str(n) + '_ws=' + str(ws),'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if i > 9:
                x.append((float(row[0]) + 1)/1000)
                y1.append(float(row[1]))
                y2.append(float(row[3]))
                y3.append(float(row[5]))
                y4.append(float(row[7]))
                y5.append(float(row[9]))
            i += 1

    plt.grid(axis='y')
    plt.grid(axis='x', color='black')

    plt.xticks(np.arange(0, (float(x[-1])), step=step), fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylim(0.45, 1)

    plt.plot(x,y1, linewidth=1.5, color='red', label='KNN-0')
    plt.plot(x,y2, linewidth=1.5, color='blue', label='KNN-100')
    plt.plot(x,y3, linewidth=1.5, color='green', label='KNN-250')
    plt.plot(x,y4, linewidth=1.5, color='mediumorchid', label='KNN-500')
    plt.plot(x,y5, linewidth=1.5, color='gold', label='KNN-750')

    plt.xlabel('thousand samples seen', fontsize=14, color='grey')
    plt.ylabel('mean\naccuracy', fontsize=14, color='grey', rotation=0)

    plt.legend()

    plt.show()
    
    

dataset_name = input("Digite o nome do dataset: ")
k_value = int(input("Digite o valor de K do KNN: "))
window_size = int(input("Digite o valor do tamanho da janela: "))
plot_step = int(input("Digite o valor do step do gráfico: "))

while(k_value > 0 and window_size > 0):
    showPlot(k_value, window_size, dataset_name, plot_step)
    dataset_name = input("Digite o nome do dataset: ")
    k_value = int(input("Digite o valor de K do KNN: "))
    window_size = int(input("Digite o valor do tamanho da janela: "))
    plot_step = int(input("Digite o valor do step do gráfico: "))

