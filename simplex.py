from importlib import simple
from sympy import Symbol, solve
from tabulate import tabulate
from utils import Table, ExpressionUtil
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
import math
class Simplex:
    """
        Inicializa a classe Simplex com a função objetivo em string.

        Args:
            string_objective_function (str): Função objetivo em string.

        Inicializa as seguintes variáveis de instância:
            - table (list): Armazena os valores da tabela simplex.
            - iterations (int): Armazena o número da iteração atual.
            - pivot_column_index (int): Armazena o índice da coluna pivô atual.
            - expression_util (ExpressionUtil): Utilitário para converter strings em listas.
            - string_objective_function (str): Função objetivo em string.
            - column_b (list): Armazena os valores da coluna b.
            - inserted (int): Armazena o número de variáveis de folga inseridas.
            - objective_function (list): Armazena a função objetivo como uma lista.
            - basic_vars (list): Armazena as variáveis básicas.
    """
    def __init__(self, string_objective_function):
        self.table = [] # armazena a tabela de valores do simplex
        self.iterations = 0 # iteração atual 
        self.pivot_column_index = 0 # coluna pivô atual
        self.expression_util = ExpressionUtil() # utilitário de expressões(str -> list)
        self.string_objective_function = string_objective_function # função objetivo em string
        self.column_b = [0] # coluna b  
        self.inserted = 0 # variáveis de folga inseridas
        self.objective_function = self._build_objective_function(string_objective_function) # função objetivo em lista
        self.basic_vars = [] # variáveis básicas 
        self.slack_vars = [] # variáveis de folga
        self.excess_vars = [] # variáveis de excesso
        self.artificial_vars = [] # variáveis artificiais
        self.first_phase = False # primeira fase
        self.entering_vars = [] # variáveis de entrada
        self.leaving_vars = [] # variáveis de saída
        


    """"Constrói a função objetivo em lista.
        Args:
            string_objective_function (str): Função objetivo em string.
        Retorna:
            list: Coeficientes da função objetivo."""
    def _build_objective_function(self, string_objective_function: str):
        variables = self.expression_util.get_variables(string_objective_function)
        row = [coef * (-1) for coef in self.expression_util.get_numeric_values(string_objective_function, fo_variables=variables)]
        return [1] + row
    
    def is_simplex_standard(self, constraint: str) -> bool:
        """Verifica se a restrição está no padrão do simplex"""
        return "<=" in constraint and self.objective == Objective.MAX.value

    def add_restriction(self, expression: str):
        delimiter = "<=" if "<=" in expression else ">=" if ">=" in expression else "="
        default_format = True
        splitted_expression = expression.split(delimiter)
        constraint = [0] + self.expression_util.get_numeric_values(
            splitted_expression[0], fo_variables=self.expression_util.get_variables(self.string_objective_function)
        )
        if not default_format:
            self.objective_function += [0]
        if delimiter == "<=":
            constraint = self.insert_slack_var(constraint, default_format)
        elif delimiter == "=":
            constraint = self.insert_artificial_var(constraint, default_format)
        elif delimiter == ">=":
            row = self.insert_artificial_var(constraint, default_format)
            constraint = self.insert_excess_var(row, default_format)
        print(f'Expressão: {expression} -> {constraint} -> {splitted_expression}')    
        
        self.column_b.append(float(splitted_expression[1]))
        self.table.append(constraint)
        print(f'Tabela: {self.table}')

        

    
    def get_entry_column(self) -> list:
        """Define a coluna pivô"""
        pivot_column = min(self.table[0])
        self.pivot_column_index = self.table[0].index(pivot_column)
        return self.pivot_column_index

    def get_pivot_line(self, entry_column: list) -> list:
        """identifica a linha que sai"""
        results = {}
        for line in range(1, len(self.table)):
            if self.table[line][entry_column-1] > 0:
                results[line] = self.table[line][-1] / self.table[line][entry_column-1]
        print(f'Coluna pivô: x{entry_column-1}, resultados: {results}')
        return min(results, key=results.get)
    
    def print_line_operation(self, row: list, pivot_line: list, new_line: list):
        row_fraction = [Fraction(value).limit_denominator() for value in row]
        pivot_line_fraction = [Fraction(value).limit_denominator() for value in pivot_line]
        new_line_fraction = [Fraction(value).limit_denominator() for value in new_line]
        for i in range(len(row)):
            print(f'{row_fraction[i]} - ({pivot_line_fraction[i]} * {pivot_line_fraction[i]}) = {new_line_fraction[i]}')


    def calculate_new_line(self, row: list, pivot_line: list) -> list:
        pivot = row[self.pivot_column_index] * -1
        result_line = [pivot * value for value in pivot_line]
        new_line = [new_value + old_value for new_value, old_value in zip(result_line, row)]
        self.print_line_operation(row, pivot_line, new_line)
        return new_line

    def calculate(self, table: list) -> None:
        self.iterations += 1

        # identifica a coluna pivô pelo menor valor
        column = self.get_entry_column()

        # linha que vai sair
        first_exit_line = self.get_pivot_line(column)

        line = table[first_exit_line]
        # identificando o pivo da linha que vai sair
        pivot = line[self.pivot_column_index]

        self.entering_vars.append(f"x{self.pivot_column_index}")
        self.leaving_vars.append(self.basic_vars[first_exit_line - 1])
        print(f'A variavel que sai da base é a {self.leaving_vars[self.iterations - 1]} e a que entra é {self.entering_vars[self.iterations - 1]}')


        # calculando nova linha pivô
        pivot_line = list(map(lambda x: x / pivot, line))
        print(f'Linha pivô: {first_exit_line}({self.basic_vars[first_exit_line -1]})')
        self.print_line_operation(line, pivot_line, pivot_line)

        # substituindo a linha que saiu pela nova linha pivô
        table[first_exit_line] = pivot_line

        # atualiza a lista de variáveis básicas
        self.basic_vars[first_exit_line - 1] = f"x{self.pivot_column_index}"

        stack = table.copy()
        line_reference = len(stack) - 1

        while stack:
            row = stack.pop()

            if line_reference != first_exit_line:
                print(f'Linha: {line_reference}({self.basic_vars[line_reference -1]})' if line_reference != 0 else 'Linha: Z')
                new_line = self.calculate_new_line(row, pivot_line)
                table[line_reference] = new_line

            line_reference -= 1
        
    def solve(self):
        self.table = Table.normalize_table(self.objective_function, self.table, self.column_b)
        self.show_table()
        while not self.is_optimal():
            self.calculate(self.table)
            self.show_table()
            
        variables = self.expression_util.get_variables(self.string_objective_function)
        return Table.get_results(self.table, variables)
    
    def is_optimal(self) -> bool:
        return min(self.table[0]) >= 0
    
    def insert_slack_var(self, row: list, default_format=True):
        """Insere variável de folga na restrição"""
        self.objective_function.append(0)
        variables = len(self.expression_util.get_variables(self.string_objective_function))
        if not self.table:
            row.append(1)
            self.inserted += 1
            self.basic_vars += [f"x{variables + self.inserted}"]
            self.slack_vars += [f"x{variables + self.inserted}"]
            return row
        
        limit = len(self.table[self.inserted - 1]) - len(row)

        for _ in range(limit):
            row.append(0)

        if not default_format:
            row = row + [-1, 1]
        else:
            row.append(1)

        self.inserted += 1
        self.basic_vars += [f"x{variables + self.inserted}"]
        self.slack_vars += [f"x{variables + self.inserted}"]
        return row
    
    def insert_artificial_var(self, row: list, default_format=True):
        """Insere variável artificial na restrição"""
        self.objective_function.append(0)
        print(row)
        variables = len(self.expression_util.get_variables(self.string_objective_function))
        if not self.table:
            row.append(1)
            self.inserted += 1
            self.basic_vars += [f"x{variables + self.inserted}"]
            self.artificial_vars += [f"x{variables + self.inserted}"]
            return row
        print(self.table, self.inserted)
        print(len(self.table))
        limit = len(self.table[self.inserted - len(self.excess_vars) - 1]) - len(row)
        for _ in range(limit):
            row.append(0)
        if not default_format:
            row = row + [1, 1]
        else:
            row.append(1)
        self.inserted += 1
        self.basic_vars += [f"x{variables + self.inserted}"]
        self.artificial_vars += [f"x{variables + self.inserted}"]
        return row

    def insert_excess_var(self, row: list, default_format=True):
        """Insere variável de excesso na restrição"""
        self.objective_function.append(0)
        variables = len(self.expression_util.get_variables(self.string_objective_function))
        if not self.table:
            row.append(1)
            self.inserted += 1
            self.basic_vars += [f"x{variables + self.inserted}"]
            self.excess_vars += [f"x{variables + self.inserted}"]
            return row
        limit = len(self.table[self.inserted - len(self.artificial_vars) - 1]) - len(row)
        for _ in range(limit):
            row.append(0)
        if not default_format:
            row = row + [1, -1]
        else:
            row.append(-1)
        self.inserted += 1
        self.basic_vars += [f"x{variables + self.inserted}"]
        self.excess_vars += [f"x{variables + self.inserted}"]
        
        return row

    def show_table(self):
        table_copy = self.table.copy()
        basic_vars = self.basic_vars.copy()
        table_copy[0] = ["Z"] + table_copy[0]
        for i in range(1, len(table_copy)):
            table_copy[i] = basic_vars[i-1:i ] + table_copy[i]
        print(f'Iteração {self.iterations}')
        print(tabulate(table_copy, tablefmt="fancy_grid", headers=["Base", "Z"] + [f'x{i}' for i in range(1, len(table_copy[0]) - 2)] + ["b"]))
    def integer_solution(self):
        pass
    
    def two_phase(self):
        num_variaveis = len(self.expression_util.get_variables(self.string_objective_function)) + self.inserted
        objective_function = self.objective_function[1:]
        constraints = self.table
        print(f'Objetivo: {objective_function}')
        print(f'Restrições: {constraints}')

        # 1a fase
        new_objective = [0] * num_variaveis
        for i in range(num_variaveis):
            if f'x{i+1}' in self.artificial_vars:
                new_objective[i] = -1
        self.objective_function =  [1] + new_objective
        print(f'Objetivo 1a fase: {self.objective_function}')
        print(self.solve())

def get_bound(x1, x2, limit):
    x_1 = [limit/x1 if x1 != 0 else limit, 0.0]
    x_2 = [0.0, limit/x2]
    return x_1, x_2


def SolucaoGrafica(problem: Simplex):
    s1 = Symbol("x1")
    s2 = Symbol("x2")

    # armazena 
    constraints = [constraint[1:problem.inserted+1] + [problem.column_b[i+1]] for i, constraint in enumerate(problem.table)]
    print(constraints)
    constraints = [(constraint[0], constraint[1], constraint[2]) for constraint in constraints]
    objective = [num * -1 for num in problem.objective_function[1:3]]

    # Inicializa listas para armazenar os limites das restrições.
    bounds_x = []
    bounds_y = []
    constraint_formulas = []

    # Get bounds (points) for constraints.
    for constraint in constraints:
        x, y = get_bound(constraint[0], constraint[1], constraint[2])
        bounds_x.append(x)
        bounds_y.append(y)
        constraint_formulas.append(constraint[0] * s1 + constraint[1] * s2 - constraint[2])

    feasible_region_points = []
    for i in range(len(constraint_formulas)):
        for j in range(i + 1, len(constraint_formulas)):
            solution = solve((constraint_formulas[i], constraint_formulas[j]), (s1, s2), dict=True)
            if solution:
                feasible_region_points.append((solution[0][s1], solution[0][s2]))

    feasible_region_points.extend([(0, 0)] + [(0, bound[1]) for bound in bounds_y if bound[1] != 0] + [(bound[0], 0) for bound in bounds_x if bound[0] != 0])

    # Filter out negative points and points that do not satisfy all constraints.
    feasible_region_points = [
        (x, y) for x, y in feasible_region_points
        if x >= 0 and y >= 0 and all(constraint.subs({s1: x, s2: y}) <= 0 for constraint in constraint_formulas)
    ]
    print(bounds_x, bounds_y)

    # Find the optimal point for the objective function.
    x1_profit, x2_profit = objective
    max_profit_formula = x1_profit * s1 + x2_profit * s2
    print([max_profit_formula.subs({'x1': point[0], 'x2': point[1]}) for point in feasible_region_points])
    optimal_point = max(feasible_region_points, key=lambda point: max_profit_formula.subs({'x1': point[0], 'x2': point[1]}))

    max_profit = float(max_profit_formula.subs({'x1': optimal_point[0], 'x2': optimal_point[1]}))

    # Plot all constraints.
    colors = ['darkgreen', 'darkblue', 'darkred', 'darkorange', 'purple']
    print(bounds_x, bounds_y, constraint_formulas)
    for i, (x, y, constraint) in enumerate(zip(bounds_x, bounds_y, constraint_formulas)):
        plt.plot(x, y, linestyle=':', marker='o', color=colors[i % len(colors)])
        plt.annotate(
            constraint,
            (x[1], y[1])
        )

    # Plot bounds for x1 >= 0 and x2 >= 0.
    plt.plot(0, 0, marker='o')
    plt.plot([0, 0], [max([b[1] for b in bounds_y]), 0], linestyle=':', color='grey')
    plt.plot([max([b[0] for b in bounds_x]), 0], [0, 0], linestyle=':', color='grey')

    feasible_region_points = sorted(feasible_region_points, key=lambda x: x[0])
    feasible_region_points.append((0, 0))
    x, y = zip(*feasible_region_points)
    plt.fill(x, y, 'lightgrey', alpha=0.5)

    # Plot point for maximum profit.
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='red')

    # plot the objective function curve
    x = np.linspace(0, 10, 100)
    y = (max_profit - objective[0] * x) / objective[1]
    plt.plot(x, y, label=f"{objective[0]}x1 + {objective[1]}x2 = {max_profit}", color='red')

    # Draw X and Y axis labels.
    plt.xlabel(f"x1", color='darkblue')
    plt.ylabel(f"x2", color='darkgreen')
    # Render plot.
    plt.show()

    

def branch_and_bound():
    """"
    1o passo) Encontrar o ótimo para o problema relaxado.
    2o passo) Avaliar: O problema é impossível? A solução é inteira?
        Se sim pare, senão...
    3o passo) Realizar partição na variável contínua xi
        onde i=1 ou i=2. Quer dizer que novos problemas,
        os descendentes, surgirão a partir da inclusão de
        novas restrições ao modelo.
    4o passo) Encontrar o ótimo para o problema relaxado e avaliar ...
    5o passo) Particionar
    6o passo) Encontrar o ótimo para o problema relaxado e avaliar
    """

if __name__ == "__main__":
    objective = "3x1 + 4x2"
    constraints = ["2x1 + 1x2 <= 600",
                   "1x1 + 1x2 <= 225",
                   "5x1 + 4x2 <= 1000",
                   "x1 + 2x2 >= 150"
                   ]
    simplex = Simplex(objective)
    for constraint in constraints:
        simplex.add_restriction(constraint)
    print(f'Problema: {objective}')
    print(f'Restrições: {constraints}')
    print(f'Variáveis de folga: {simplex.slack_vars}')
    print(f'Variáveis básicas: {simplex.basic_vars}')
    print(f'Variáveis artificiais: {simplex.artificial_vars}')
    print(f'Variáveis de excesso: {simplex.excess_vars}')
    
    
    # SolucaoGrafica(simplex)
    # print(simplex.solve())
    simplex.two_phase()
    # LinearProgrammingPlotter.plot_solution(objective, constraints)
