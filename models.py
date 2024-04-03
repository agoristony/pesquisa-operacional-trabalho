from constants import *

class Variavel:
    def __init__(self, nome: str, limiteInferior: float = -INFINITO, limiteSuperior: float = INFINITO, categoria: str = CONTINUO):
        self.nome = nome
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior
        self.categoria = categoria
    
    def __str__(self):
        return self.nome

class Restricao:
    def __init__(self, nome: str, variaveis: list[Variavel], operador: int = IGUAL, valor: float = 0):
        self.nome = nome
        self.variaveis = variaveis
        self.operador = operador
        self.valor = valor
    
    def __str__(self):
        return self.nome


class Problema:
    def __init__(self, nome: str, variaveis: list[Variavel], restricoes: list[Restricao], modo: int = MINIMIZAR):
        self.nome = nome
        self.variaveis = variaveis
        self.restricoes = restricoes
        self.funcaoObjetivo = None
        self.modo = modo

    def __str__(self):
        return self.nome
    
    def funcao_objetivo(self, funcao: str):
        self.funcaoObjetivo = funcao


class Solver:
    def __init__(self, problema: Problema, metodo: str):
        self.problema = problema
        self.metodo = metodo
        print(f"Solver criado com o método {metodo} para o problema {problema.nome}")



    
