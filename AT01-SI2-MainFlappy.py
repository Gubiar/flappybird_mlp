# Importando as classes necessárias
from Demo import FlappyBird_Human
from MLP import MLP
from GeneticAlgorithm import GeneticAlgorithm

def main():
    # Definindo os parâmetros da MLP e do algoritmo genético
    entrada = 2  # número de entradas (por exemplo, distância vertical e horizontal até o próximo tubo)
    oculta = 5  # número de neurônios na camada oculta
    saida = 1  # número de saídas (por exemplo, pular ou não pular)
    taxaDeAprendizado = 0.1  # taxa de aprendizado da MLP

    # Inicializando o algoritmo genético
    ga = GeneticAlgorithm(entrada, oculta, saida, taxaDeAprendizado)

    # Executando o algoritmo genético
    ga.execute()

    # Imprimindo a melhor aptidão obtida
    print("Melhor aptidão:", ga.bestFitness)

if __name__ == "__main__":
    main()
