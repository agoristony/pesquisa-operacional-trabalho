from atexit import register
from pyparsing import col
from tabulate import tabulate
from utils import Table, ExpressionUtil, graphical_solution, primal_to_dual
from fractions import Fraction
import matplotlib.pyplot as plt


import enum

class Objective(enum.Enum):
    MAX = 0
    MIN = 1

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
    def __init__(self, string_objective_function, objective = Objective.MAX.value):
        self.table = [] # armazena a tabela de valores do simplex
        self.iterations = 0 # iteração atual 
        self.iteration_history = [] # histórico de iterações
        self.pivot_column_index = 0 # coluna pivô atual
        self.expression_util = ExpressionUtil() # utilitário de expressões(str -> list)
        self.string_objective_function = string_objective_function # função objetivo em string
        self.column_b = [0] # coluna b  
        self.inserted = 0 # variáveis de folga inseridas
        self.objective = objective # objetivo da função
        self.basic_vars = [] # variáveis básicas 
        self.slack_vars = [] # variáveis de folga
        self.excess_vars = [] # variáveis de excesso
        self.artificial_vars = [] # variáveis artificiais
        self.entering_vars = [] # variáveis de entrada
        self.leaving_vars = [] # variáveis de saída
        self.first_phase = False
        
        if self.objective == Objective.MAX.value:
            self.objective_function = self._build_objective_function(string_objective_function, True)
        else:
            self.objective_function = self._build_objective_function(string_objective_function, False)
    
    def __repr__(self):
        return f'Z = {self.string_objective_function} \nSujeito a: {self.constraints}'

    """"Constrói a função objetivo em lista.
        Args:
            string_objective_function (str): Função objetivo em string.
        Retorna:
            list: Coeficientes da função objetivo."""
    def _build_objective_function(self, string_objective_function: str, maximize: bool) -> list:
        variables = self.expression_util.get_variables(string_objective_function)
        row = self.expression_util.get_numeric_values(string_objective_function, fo_variables=variables)
        if maximize:
            row = [coef * -1 for coef in row]
        print(f'Função objetivo: {row}')
        return row
    
    def is_simplex_standard(self, constraint: str) -> bool:
        """Verifica se a restrição está no padrão do simplex"""
        return "<=" in constraint and self.objective == Objective.MAX.value

    def add_restriction(self, expression: str):
        delimiter = "<=" if "<=" in expression else ">=" if ">=" in expression else "="
        default_format = True
        splitted_expression = expression.split(delimiter)
        constraint = self.expression_util.get_numeric_values(
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
        
        self.column_b.append(float(splitted_expression[1]))
        self.table.append(constraint)
        return constraint, splitted_expression[1]

    
    def get_entry_column(self) -> list:
        """Define a coluna pivô"""
        pivot_column = min(self.table[0][:-1]) if self.objective == Objective.MAX.value else max(self.table[0][:-1])
        last_index_of_ocurrence = None
        for i, value in enumerate(self.table[0][:-1]):
            if value == pivot_column:
                last_index_of_ocurrence = i
        print(f'Coluna pivô: {pivot_column}, index: {self.table[0][:-1].index(pivot_column)}, {self.get_all_vars()}')
        self.pivot_column_index = self.table[0][:-1].index(pivot_column)
        return self.pivot_column_index

    def get_pivot_line(self, entry_column: list) -> list:
        """identifica a linha que sai"""
        results = {}
        for line in range(1, len(self.table)):
            if self.table[line][entry_column] > 0:
                results[line] = self.table[line][-1] / self.table[line][entry_column]
            elif self.objective == Objective.MIN.value:
                try:
                    results[line] = self.table[line][-1] / self.table[line][entry_column]
                except ZeroDivisionError:
                    results[line] = float('inf')
        if not results:
            raise Exception(f"Nenhum valor positivo encontrado para a coluna pivô - {entry_column}")

        results = {key: value for key, value in results.items() if value > 0}
        return min(results, key=results.get)
    
    def print_line_operation(self, row: list, pivot_line: list, new_line: list, pivot: int = 0):
        row_fraction = [Fraction(value).limit_denominator() for value in row]
        pivot_line_fraction = [Fraction(value).limit_denominator() for value in pivot_line]
        new_line_fraction = [Fraction(value).limit_denominator() for value in new_line]
        
        try:
            for i in range(len(row)):
                print(f'{row_fraction[i]} - ({pivot} * {pivot_line_fraction[i]}) = {new_line_fraction[i]}')
        except:
            print(f'row: {len(row)} row_fraction: {len(row_fraction)}')
            print(f'Tamanhos: {len(row_fraction)} - {len(pivot_line_fraction)} - {len(new_line_fraction)}')
            raise Exception(f'Erro ao imprimir a operação. Vetores: {row} - {pivot_line} - {new_line}')

    def calculate_new_line(self, row: list, pivot_line: list, verbose) -> list:
        
        if len(row) != len(pivot_line):
            row = row[:-1]
        pivot = row[self.pivot_column_index] * -1 
        result_line = [pivot * value for value in pivot_line]
        new_line = [round(new_value + old_value,6) for new_value, old_value in zip(result_line, row)]
        if verbose:
            print(f'Calculando nova linha: {row} - {pivot_line}')
            self.print_line_operation(row, pivot_line, new_line, pivot)
        return new_line

    def calculate(self, table: list, verbose, column=None, first_exit_line=None):
        """
            Calcula a próxima iteração do simplex
            - Identifica a coluna pivô
            - Identifica a linha que sai
            - Calcula a nova linha pivô
            - Substitui a linha que saiu pela nova linha pivô
            - Atualiza a lista de variáveis básicas
        """
        

        if not column:
            column = self.get_entry_column()
        if not first_exit_line:
            first_exit_line = self.get_pivot_line(column)
        print(f'Coluna: {column} - Linha: {first_exit_line}')
        self.iteration_history.append(self.register_iteration(column, first_exit_line))
        self.iterations += 1
        line = table[first_exit_line]
        pivot = line[self.pivot_column_index]
        self.entering_vars.append(f"{self.get_all_vars()[self.pivot_column_index]}")
        self.leaving_vars.append(self.basic_vars[first_exit_line - 1])
        print(f'A variavel que sai da base é a {self.leaving_vars[self.iterations - 1]} e a que entra é {self.entering_vars[self.iterations - 1]}')


        # nova linha pivô
        pivot_line = list(map(lambda x: x / pivot, line))
        print(f'Linha pivô: {first_exit_line}({self.basic_vars[first_exit_line -1]})')
        if verbose:
            self.print_line_operation(line, pivot_line, pivot_line)

        # substituindo a linha que saiu pela nova linha pivô
        table[first_exit_line] = pivot_line

        self.basic_vars[first_exit_line - 1] = f"{self.get_all_vars()[self.pivot_column_index]}"
        stack = table.copy()
        line_reference = len(stack) - 1

        # atualiza as demais linhas
        while stack:
            row = stack.pop()
            if line_reference != first_exit_line:
                if verbose:
                    print(f'Linha: {line_reference}({self.basic_vars[line_reference -1]})' if line_reference != 0 else 'Linha: Z')
                new_line = self.calculate_new_line(row, pivot_line, verbose)
                table[line_reference] = new_line
            line_reference -= 1
    
    def get_results(self):
        """Retorna os resultados do problema"""
        results = {}
        results['solucao'] = self.table[0][-1]
        for i in range(1, len(self.table)):
            results[self.basic_vars[i-1]] = self.table[i][-1]
        return results
        
    def register_iteration(self, pivot_column=None, pivot_line=None):
        iteracao = {
            'iteracao': self.iterations,
            'table': self.table.copy(),
            'basic_vars': self.basic_vars.copy(),
            'header_vars': self.get_all_vars(),
            'pivot_column_index': pivot_column,
            'pivot_line_index': pivot_line,
        }
        print(f'Iteração: {iteracao}')
        return iteracao
    
    def solve(self, verbose=True):
        """Resolve o problema de programação linear"""    
        response = {
            'iteracoes':[],
            'solucao': None,
            'tipo': 'max' if self.objective == Objective.MAX.value else 'min',
            'tipo_problema': 'duas_fases' if self.artificial_vars != [] else 'simplex'
        }
        
        if self.artificial_vars != [] and not self.first_phase:
            return self.two_phase(verbose) 
        self.show_table()   
        
        while not self.is_optimal():
            self.calculate(self.table, verbose)
            if verbose:
                self.show_table()
            if self.iterations > 50:
                print('Número máximo de iterações atingido')
                break
        variables = self.expression_util.get_variables(self.string_objective_function)
        response['solucao'] = self.get_results()
        self.iteration_history.append(self.register_iteration(None, None))
        response['iteracoes'] = self.iteration_history 
        return response
    
    def is_optimal(self) -> bool:
        """verifica se o valor da função objetivo é otimo para o problema"""
        num_vars = len(self.expression_util.get_variables(self.string_objective_function))
        if self.objective == Objective.MAX.value:
            return all(round(value, 2) >= 0 for value in self.table[0][:-1])
        else:
            return all(round(value, 2) <= 0 for value in self.table[0][:num_vars])
    
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
        
        print(f'{self.inserted} - {self.table} - {row}')
        limit = len(self.table[-1]) - len(row)

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

        variables = len(self.expression_util.get_variables(self.string_objective_function))
        if not self.table:
            row.append(1)
            self.inserted += 1
            self.basic_vars += [f"x{variables + self.inserted}"]
            self.artificial_vars += [f"x{variables + self.inserted}"]
            return row

        limit = len(self.table[self.inserted - len(self.excess_vars) - 1]) - len(row)
        for _ in range(limit):
            row.append(0)
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
            row.append(-1)
            self.inserted += 1
            self.excess_vars += [f"x{variables + self.inserted}"]
            return row
        limit = len(self.table[self.inserted - len(self.artificial_vars) - 1]) - len(row)
        for _ in range(limit):
            row.append(0)
        row.append(-1)
        self.inserted += 1
        self.excess_vars += [f"x{variables + self.inserted}"]
        return row

    def get_all_vars(self):
        """Retorna todas as variáveis do problema, incluindo as de folga, excesso e artificiais"""
        variables = self.expression_util.get_variables(self.string_objective_function)
        variables += [*self.slack_vars, *self.artificial_vars, *self.excess_vars]
        return sorted(variables)

    def show_table(self):
        """Exibe a tabela simplex"""
        table_copy = self.table.copy()
        basic_vars = self.basic_vars.copy()
        table_copy[0] = ["Z"] + table_copy[0]
        for i in range(1, len(table_copy)):
            table_copy[i] = basic_vars[i-1:i ] + table_copy[i]
        print(f'Iteração {self.iterations}')
        print(tabulate(table_copy, tablefmt="fancy_grid", headers=["Base"] + [*self.get_all_vars()] + ["b"]))
    
    def two_phase(self, verbose= True):
        """Resolve o problema de programação linear utilizando o método das duas fases"""
        num_variaveis =  len(self.expression_util.get_variables(self.string_objective_function)) + self.inserted
        # 1a fase
        
        new_objective = [0] * num_variaveis
        for i in range(num_variaveis):
            if f'x{i+1}' in self.artificial_vars:
                new_objective[i] = -1 if self.objective == Objective.MAX.value else 1
            else:
                new_objective[i] = 0
        self.objective_function =  new_objective + [0]
        
        for var in self.artificial_vars:
            index = self.basic_vars.index(var)
            row = self.table[index+1]
            for i in range(len(row)):
                self.objective_function[i] += row[i]
            
        self.table[0] = self.objective_function
        original_objective = self.objective
        self.objective = Objective.MIN.value
        self.first_phase = True
        solucao = self.solve()
        to_remove = []
        for i in range(len(row)):
            if f'x{i+1}' in self.artificial_vars:
                to_remove.append(i)
                self.inserted -= 1
        for i in range(len(self.table)):
            self.table[i] = [value for j, value in enumerate(self.table[i]) if j not in to_remove]
        self.artificial_vars = []
        
        # 2a fase
        print('2a fase')
        self.first_phase = False
        self.objective_function = self._build_objective_function(self.string_objective_function, True if original_objective == Objective.MAX.value else False)
        
        for i in range(self.inserted):
            self.objective_function.append(0)
        self.table[0] = self.objective_function + [0]
        self.objective = original_objective
        solucao['iteracoes'].append(self.register_iteration(self.pivot_column_index, self.get_pivot_line(self.pivot_column_index)))
        solucao_fase_2 = self.solve()
        solucao['solucao'] = solucao_fase_2['solucao']
        
        return solucao
    

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
    objective = "400x1 + 600x2 + 900x3 + 350x4"
    constraints = ["200x2 + 300x3 <= 14000",
                   "20000x2+400000x3+50000x4 <= 12750000",
                   "1x1 + 1x2 + 1x3 + 1x4 = 100",
                   "x4 >= 20",
                   ]
    
    objective = "3x1 + 2x2"
    constraints = ["1x1 + 2x2 <= 10",
                   "3x1 + 1x2 >= 15",
                   "1x1 + 1x2 = 7",
                   ]
    
    
    # objective = "5x1 + 4x2"
    # constraints = ["3x1 + 2x2 >= 5",
    #                "2x1 + 3x2 >= 7",
                #    "x2 >= 1",
                #    ]
    
    objective = "x1 + x2 + x3"
    constraints = ["0.8x1 + 0.7x2 + 0.9x3 >= 30",
                     "0.6x1 + 0.5x2 + 0.8x3 >= 25",
                     "0.4x1 + 0.5x2 + 0.3x3 <= 20",
                     ]

    # objective = "5x1 + 6x2"
    # constraints = ["x1 + x2 <= 5",
    #                "4x1 + 7x2 <= 28",
    #                ]
    simplex = Simplex(objective, Objective.MAX.value)
    for constraint in constraints:
        simplex.add_restriction(constraint)
    simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
    print(simplex.table)
    print(f'Problema: {objective}')
    print(f'Restrições: {constraints}')
    
    # print(primal_to_dual(simplex))
    # graphical_solution(simplex)
    print(simplex.solve())

    # simplex.two_phase()
    # LinearProgrammingPlotter.plot_solution(objective, constraints)
