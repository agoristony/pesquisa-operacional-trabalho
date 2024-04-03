from ast import List
import string
from constants import *
import sympy as sym

class Variavel:
    def __init__(self, nome: str, limiteInferior: float = -INFINITO, limiteSuperior: float = INFINITO, categoria: str = CONTINUO):
        self.nome = nome
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior
        self.categoria = categoria
    
    def __str__(self):
        return self.nome

class Restricao:
    def __init__(self, nome: str, variaveis: list[Variavel], coeficientes_variaveis: list[float] = [], operador: int = IGUAL, valor: float = 0):
        self.nome = nome
        self.variaveis = variaveis
        self.coeficientes_variaveis = coeficientes_variaveis
        self.operador = operador
        self.valor = valor
    
    def __str__(self):
        string = f"{self.nome}: "
        for i in range(len(self.variaveis)):
            string += f"{self.coeficientes_variaveis[i]}{self.variaveis[i]}"
            if i < len(self.variaveis) - 1:
                string += " + "
        string += f" {OPERADORES[self.operador]} {self.valor}"
        return string


class Problema:
    def __init__(self, nome: str, variaveis: list[Variavel], restricoes: list[Restricao], modo: int = MINIMIZAR):
        self.nome = nome
        self.variaveis = variaveis
        self.restricoes = restricoes
        self.funcaoObjetivo = []
        self.modo = modo

    def __str__(self):
        return self.nome
    
    def funcao_objetivo(self):
        # imprime a funcao no formato ax1 + bx2 + cx3 + ... + kxn
        coeficientes = self.funcaoObjetivo
        symbols = [sym.Symbol(f"x{i}") for i in range(len(coeficientes))]
        funcao = sum([coeficientes[i] * symbols[i] for i in range(len(coeficientes))])
        return f'{funcao}'
    
        

class Solver:
    def __init__(self, problema: Problema, metodo: str):
        self.problema = problema
        self.metodo = metodo
        print(f"Solver criado com o método {metodo} para o problema {problema.nome}")

    def solucaoGrafica(self):
        # encontrar a solução do problema pelo método gráfico
        # zerar uma variável e encontrar o valor da outra
        for restricao in self.problema.restricoes:
            pass

    
