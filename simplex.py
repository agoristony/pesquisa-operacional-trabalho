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
    

    def add_restriction(self, expression: str):
        delimiter = "<="
        default_format = True

        # if not self.is_simplex_standard(expression):
        #     raise ValueError("Simplex Duas Fases não implementado!")

        splitted_expression = expression.split(delimiter)
        constraint = [0] + self.expression_util.get_numeric_values(
            splitted_expression[0], fo_variables=self.expression_util.get_variables(self.string_objective_function)
        )
        if not default_format:
            self.objective_function += [0]
        if delimiter == "<=":
            constraint = self.insert_slack_var(constraint, default_format)
        self.column_b.append(float(splitted_expression[1]))
        self.table.append(constraint)

    def get_dual(self):
        # Extrai o número de variáveis primais e restrições
        num_primal_vars = len(self.objective_function) - 1
        num_constraints = len(self.table)

        # Prepara a função objetivo dual (coeficientes são os valores da coluna b)
        dual_obj_coeffs = [str(self.column_b[i]) + "x" + str(i+1) for i in range(num_constraints)]
        dual_obj_function = " + ".join(dual_obj_coeffs)

        # Criar uma nova instância de Simplex para representar o problema dual
        dual_simplex = Simplex(dual_obj_function)

        # Transpor a tabela para obter os coeficientes das restrições
        for j in range(num_primal_vars):
            dual_constraint_coeffs = [self.table[i][j+1] for i in range(num_constraints)]  # Pula o primeiro elemento (coeficiente da variável)
            rhs_value = self.objective_function[j+1]  # Coeficiente da variável na função objetivo
            dual_constraint_expr = " + ".join([f"{coeff}x{i+1}" for i, coeff in enumerate(dual_constraint_coeffs)])
            dual_constraint = f"{dual_constraint_expr} <= {rhs_value}"
            dual_simplex.add_restriction(dual_constraint)

        return dual_simplex
    
    def get_entry_column(self) -> list:
        """Define a coluna pivô"""
        pivot_column = min(self.table[0])
        self.pivot_column_index = self.table[0].index(pivot_column)
        return self.pivot_column_index

    def get_pivot_line(self, entry_column: list) -> list:
        """identifica a linha que sai"""
        results = {}
        for line in range(1, len(self.table)):
            if self.table[line][entry_column] > 0:
                results[line] = self.table[line][-1] / self.table[line][entry_column]
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
        return row
    
    def show_table(self):
        table_copy = self.table.copy()
        basic_vars = self.basic_vars.copy()
        table_copy[0] = ["Z"] + table_copy[0]
        for i in range(1, len(table_copy)):
            table_copy[i] = basic_vars[i-1:i ] + table_copy[i]
        print(f'Iteração {self.iterations}')
        print(tabulate(table_copy, tablefmt="fancy_grid", headers=["Base", "Z"] + [f"x{i}" for i in range(1, len(table_copy[0]) - 2)] + ["b"]))
    
    def integer_solution(self):
        pass
    
class LinearProgrammingPlotter:
    @staticmethod
    def plot_solution(objective_function, constraints):
        obj_coeffs = LinearProgrammingPlotter.parse_expression(objective_function)
        
        max_x1 = max_x2 = 0
        for constraint in constraints:
            parts = constraint.split(' ')
            coeffs, rhs = LinearProgrammingPlotter.parse_expression(parts[0]), float(parts[-1])
            max_x1 = max(max_x1, rhs / abs(coeffs[0]) if coeffs[0] != 0 else 0)
            max_x2 = max(max_x2, rhs / abs(coeffs[1]) if coeffs[1] != 0 else 0)

        max_x1, max_x2 = math.ceil(max_x1) + 10, math.ceil(max_x2) + 10
        
        x1 = np.linspace(0, max_x1, 400)
        x2 = np.linspace(0, max_x2, 400)
        X1, X2 = np.meshgrid(x1, x2)

        fig, ax = plt.subplots()
        feasible_set = np.zeros(X1.shape, dtype=bool)
        
        for constraint in constraints:
            parts = constraint.split(' ')
            coeffs, rhs = LinearProgrammingPlotter.parse_expression(parts[0]), float(parts[-1])
            if '<=' in constraint:
                condition = coeffs[0] * X1 + coeffs[1] * X2 <= rhs
            elif '>=' in constraint:
                condition = coeffs[0] * X1 + coeffs[1] * X2 >= rhs
            else:
                continue 
            
            feasible_set |= condition
            ax.contour(X1, X2, coeffs[0] * X1 + coeffs[1] * X2 - rhs, levels=0, colors='k')
        
        ax.imshow(feasible_set, extent=(X1.min(), X1.max(), X2.min(), X2.max()), origin='lower', alpha=0.3)

        z_func = lambda x1, z: (z - obj_coeffs[0] * x1) / obj_coeffs[1]
        z_values = [min(obj_coeffs) * 20, max(obj_coeffs) * 20]
        for z in z_values:
            ax.plot(x1, z_func(x1, z), 'r--')

        ax.set_xlim([0, max_x1])
        ax.set_ylim([0, max_x2])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(objective_function)
        plt.show()

    @staticmethod
    def parse_expression(expression):
        terms = expression.replace('-', '+-').split('+')
        coeffs = [0, 0]
        for term in terms:
            if 'x1' in term:
                coeffs[0] = float(term.replace('x1', ''))
            elif 'x2' in term:
                coeffs[1] = float(term.replace('x2', ''))
        return coeffs


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
    objective = "1x1 + 4x2"
    constraints = ["-2x1 + 4x2 <= 8", "2x1 + 3x2 <= 12"]
    simplex = Simplex(objective)
    for constraint in constraints:
        simplex.add_restriction(constraint)
    print(simplex.solve())
    # print(simplex.get_dual().solve())
    LinearProgrammingPlotter.plot_solution(objective, constraints)
