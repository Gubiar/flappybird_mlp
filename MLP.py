import numpy as np

class MLP(object):
    def __init__(self, entrada, oculta, saida, taxaDeAprendizado = 0.1):
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.taxaDeAprendizado = taxaDeAprendizado

        # Inicializa os pesos com valores aleatórios
        self.pesos_entrada_oculta = np.random.randn(self.oculta, self.entrada)
        self.pesos_oculta_saida = np.random.randn(self.saida, self.oculta)

    def getTaxaDeAprendizado(self):
        return self.taxaDeAprendizado

    def setTaxaDeAprendizado(self,taxa):
        self.taxaDeAprendizado = taxa

    def ativacaoSigmoidal(self, valor):
        return 1 / (1 + np.exp(-valor))

    def derivadaAtivacaoSigmoidal(self, valor):
        return valor * (1 - valor)

    def erroQuadraticoMedio(self, esperado, valor):
        return np.mean(np.square(esperado - valor))

    def feedForward(self, dados):
        self.camada_entrada = dados
        self.camada_oculta = self.ativacaoSigmoidal(np.dot(self.pesos_entrada_oculta, self.camada_entrada))
        self.camada_saida = self.ativacaoSigmoidal(np.dot(self.pesos_oculta_saida, self.camada_oculta))
        return self.camada_saida

    def backPropagation(self, esperado):
        erro_saida = esperado - self.camada_saida
        delta_saida = erro_saida * self.derivadaAtivacaoSigmoidal(self.camada_saida)

        erro_oculta = np.dot(self.pesos_oculta_saida.T, delta_saida)
        delta_oculta = erro_oculta * self.derivadaAtivacaoSigmoidal(self.camada_oculta)

        # Atualiza os pesos
        self.pesos_oculta_saida += self.taxaDeAprendizado * np.dot(delta_saida[:,None], np.transpose(self.camada_oculta[:,None]))
        self.pesos_entrada_oculta += self.taxaDeAprendizado * np.dot(delta_oculta[:,None], np.transpose(self.camada_entrada[:,None]))

    def treinamento(self, dados, esperado):
        # Faz a classificação
        self.feedForward(dados)
        
        # Faz a correção dos pesos
        self.backPropagation(esperado)
