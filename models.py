from dataclasses import dataclass
from constants import *

class Variavel:
    def __init__(self, nome: str, limiteInferior: float = -INFINITO, limiteSuperior: float = INFINITO, categoria: str = CONTINUO):
        self.nome = nome
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior
        self.categoria = categoria

class Restricao:
    def __init__(self, nome: str, variaveis: list[Variavel], operador: int = IGUAL, valor: float = 0):
        self.nome = nome
        self.variaveis = variaveis
        self.operador = operador
        self.valor = valor


class Problema:
    def __init__(self, nome: str, variaveis: list[Variavel], restricoes: list[Restricao]):
        self.nome = nome
        self.variaveis = variaveis
        self.restricoes = restricoes
        self.funcaoObjetivo = None


    def __str__(self):
        return self.nome
    
    def funcao_objetivo(self, funcao: str):
        self.funcaoObjetivo = funcao

    


    