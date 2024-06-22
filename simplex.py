from matplotlib import pyplot as plt
from sympy import Symbol, solve
from tabulate import tabulate
from utils import Table, ExpressionUtil
from fractions import Fraction
import numpy as np


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
        self.restriction_strings = [] # restrições em string
        self.first_phase = False
        
        self.objective_function = self._build_objective_function(string_objective_function)
    
    def __repr__(self):
        return f'Z = {self.string_objective_function} \nSujeito a: {self.restriction_strings}'

    """"Constrói a função objetivo em lista.
        Args:
            string_objective_function (str): Função objetivo em string.
        Retorna:
            list: Coeficientes da função objetivo."""
    def _build_objective_function(self, string_objective_function: str) -> list:
        variables = self.expression_util.get_variables(string_objective_function)
        row = self.expression_util.get_numeric_values(string_objective_function, fo_variables=variables)
        print(f'Função objetivo: {row}')
        return row
    
    def is_simplex_standard(self, constraint: str) -> bool:
        """Verifica se a restrição está no padrão do simplex"""
        return "<=" in constraint and self.objective == Objective.MAX.value
    
    def get_constraint_from_string(self, expression: str) -> list:
        """Retorna a restrição em formato de lista"""
        delimiter = "<=" if "<=" in expression else ">=" if ">=" in expression else "="
        splitted_expression = expression.split(delimiter)
        constraint = self.expression_util.get_numeric_values(
            splitted_expression[0], fo_variables=self.expression_util.get_variables(self.string_objective_function)
        )
        return constraint, delimiter, splitted_expression

    def add_restriction(self, expression: str):
        '''Adiciona uma restrição ao problema de programação linear
        Args:
            expression (str): Restrição em string.
        Retorna:
            list: Restrição em formato de lista.
            
        '''
        self.restriction_strings.append(expression)
        default_format = True
        constraint, delimiter, splitted_expression = self.get_constraint_from_string(expression)
        if not default_format:
            self.objective_function += [0]
        if delimiter == "<=":
            constraint = self.insert_slack_var(constraint, default_format)
        elif delimiter == "=":
            constraint = self.insert_artificial_var(constraint, default_format, delimiter)
        elif delimiter == ">=":
            row = self.insert_artificial_var(constraint, default_format, delimiter)
            constraint = self.insert_excess_var(row, default_format)
        
        self.column_b.append(float(splitted_expression[1]))
        self.table.append(constraint)
        return constraint, splitted_expression[1]

    
    def get_entry_column(self) -> list:
        """Define a coluna pivô"""
        pivot_column = min(self.table[0][:-1]) if self.objective == Objective.MIN.value else max(self.table[0][:-1])
        pivot_index = None
        for index, value in enumerate(self.table[0][:-1]):
            if value == pivot_column:
                pivot_index = index
        print(f'Coluna pivô: {pivot_column}, index: {pivot_index}, {self.get_all_vars()}')
        self.pivot_column_index = pivot_index
        return self.pivot_column_index

    def get_pivot_line(self, entry_column: list) -> list:
        """identifica a linha que sai"""
        results = {}
        for line in range(1, len(self.table)):
            if self.table[line][entry_column] > 0:
                results[line] = self.table[line][-1] / self.table[line][entry_column]
                print(f'Valor: {self.table[line][-1]} / {self.table[line][entry_column]}')
            elif self.objective == Objective.MIN.value:
                try:
                    results[line] = self.table[line][-1] / self.table[line][entry_column]
                except ZeroDivisionError:
                    results[line] = float('inf')
        if not results:
            print(f'not results Linhas: {results}')
            raise Exception(f"Nenhum valor positivo encontrado para a coluna pivô {entry_column} - Problema ilimitado")
        results = {key: value for key, value in results.items() if value >= 0}
        return min(results, key=results.get)
    
    def print_line_operation(self, row: list, pivot_line: list, new_line: list, pivot: int = 0):
        '''Imprime a operação realizada na linha do simplex'''
        row_fraction = [Fraction(value).limit_denominator(2) for value in row]
        pivot_line_fraction = [Fraction(value).limit_denominator(2) for value in pivot_line]
        new_line_fraction = [Fraction(value).limit_denominator(2) for value in new_line]
        operation = []
        try:
            for i in range(len(row)):
                operation.append(f'{row_fraction[i]} - ({round(pivot,3)} * {pivot_line_fraction[i]}) = {new_line_fraction[i]}')
            return operation
        except:
            raise Exception(f'Erro ao imprimir a operação. Vetores: {row} - {pivot_line} - {new_line}')

    def calculate_new_line(self, row: list, pivot_line: list, verbose) -> list:
        '''Calcula a nova linha da tabela simplex a partir da linha pivô'''
        if len(row) != len(pivot_line):
            row = row[:-1]
        pivot = row[self.pivot_column_index] * -1 
        result_line = [pivot * value for value in pivot_line]
        new_line = [new_value + old_value for new_value, old_value in zip(result_line, row)]
        operation = self.print_line_operation(row, pivot_line, new_line, pivot)
        return new_line, operation

    def calculate(self, table: list, verbose, column=None, first_exit_line=None):
        """
        Args:
            table (list): Tabela simplex.
            verbose (bool): Exibe a tabela simplex.
            column (list): Coluna pivô.
            first_exit_line (list): Linha que sai.
        Retorna:
            None
            
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
            try:
                first_exit_line = self.get_pivot_line(column)
            except Exception as e:
                return 'Problema ilimitado'
        print(f'Coluna: {column} - Linha: {first_exit_line}')
        
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
        operations = []
        while stack:
            row = stack.pop()
            if line_reference != first_exit_line:
                new_line, operation = self.calculate_new_line(row, pivot_line, verbose)
                operations.append([f'Linha: {line_reference}({self.basic_vars[line_reference -1]})' if line_reference != 0 else 'Linha: Z'] + operation)
                table[line_reference] = new_line
            line_reference -= 1
        self.iteration_history.append(self.register_iteration(column, first_exit_line, operations))
        self.iterations += 1
            
    
    def get_results(self):
        """Retorna os resultados do problema"""
        results = {}
        for i in range(1, len(self.table)):
            results[self.basic_vars[i-1]] = round(self.table[i][-1],3)
        results['solucao'] = round(self.table[0][-1] * -1, 3)
        print(f'Solução: {results}')
        return results
        
    def register_iteration(self, pivot_column=None, pivot_line=None, operations=None):
        '''Registra a iteração atual do simplex para históricp'''
        iteracao = {
            'iteracao': self.iterations,
            'table': self.table.copy(),
            'basic_vars': self.basic_vars.copy(),
            'header_vars': self.get_all_vars(),
            'pivot_column_index': pivot_column,
            'pivot_line_index': pivot_line,
            'operacoes': operations,
        }
        return iteracao
    
    def basic_vars_indexes(self):
        """Retorna os índices das variáveis básicas"""
        indexes = []
        for var in self.basic_vars:
            indexes.append(self.get_all_vars().index(var))
        return indexes
    
    def solve(self, verbose=True):
        """Resolve o problema de programação linear"""    
        response = {
            'iteracoes':[],
            'solucao': None,
            'tipo': 'max' if self.objective == Objective.MAX.value else 'min',
            'tipo_problema': 'duas_fases' if self.artificial_vars != [] else 'simplex',
            'variaveis_nao_basicas': [],
            'solucao_inteira': False,
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
        response['solucao'] = self.get_results()
        self.iteration_history.append(self.register_iteration(None, None, None))
        response['iteracoes'] = self.iteration_history 
        response['variaveis_nao_basicas'] = {value:0 for key, value in enumerate(self.get_all_vars()) if value not in self.basic_vars}
        response['solucao_inteira'] = all(round(value, 3).is_integer() for key, value in response['solucao'].items() if key in self.expression_util.get_variables(self.string_objective_function))
        return response
    
    def is_optimal(self) -> bool:
        """verifica se o valor da função objetivo é otimo para o problema"""
        if self.first_phase:
            basic_vars_indexes = self.basic_vars_indexes()
            return all(round(value, 3) >= 0 for value in self.table[0][:-1]) and all(round(self.table[0][i], 3) == 0 for i in range(len(self.table[0])) if i in basic_vars_indexes)
        elif self.objective == Objective.MAX.value:
            return all(round(value, 3) <= 0 for value in self.table[0][:-1])
        return all(round(value, 3) >= 0 for value in self.table[0][:-1])
    
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
    
    def insert_artificial_var(self, row: list, default_format=True, delimiter=None):
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
        if delimiter == ">=" or ((delimiter == "=") and self.objective == Objective.MIN.value):
            row.append(1)
        elif delimiter == "=":
            row.append(0)
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
    
    def dual(self):
        """Converte o problema primal em dual
        Se o problema primal é de maximização, o dual é de minimização e vice-versa
        Retorna:
            Simplex: Instância da classe Simplex com o problema dual.
        """
        table = self.table
        n_vars = len(self.expression_util.get_variables(self.string_objective_function))
        new_objective = []
        new_restriction_strings = []
        for i in range(n_vars):
            new_restriction = [restr[i] for restr in table[1:]] + [table[0][i]]
            _, delimiter, _ = self.get_constraint_from_string(self.restriction_strings[i])
            new_objective.append(self.column_b[i+1])
            if self.objective == Objective.MAX.value:
                if delimiter == ">=":
                    new_restriction = [-value for value in new_restriction]
                    new_objective[-1] = -new_objective[-1]
                    delimiter = "<="
                    new_restriction_strings.append(f"{'+'.join([str(value) + f'x{i+1}' for i, value in enumerate(new_restriction[:-1])])} <= {new_restriction[-1]}")
                else:
                    new_restriction_strings.append(f"{'+'.join([str(value) + f'x{i+1}' for i, value in enumerate(new_restriction[:-1])])} >= {new_restriction[-1]}")
            else:
                if delimiter == "<=":
                    new_restriction = [-value for value in new_restriction]
                    new_objective[-1] = -new_objective[-1]
                    delimiter = ">="
                    new_restriction_strings.append(f"{'+'.join([str(value) + f'x{i+1}' for i, value in enumerate(new_restriction[:-1])])} >= {new_restriction[-1]}")
                else:
                    new_restriction_strings.append(f"{'+'.join([str(value) + f'x{i+1}' for i, value in enumerate(new_restriction[:-1])])} <= {new_restriction[-1]}")
        print(f'Objetivo: {new_objective}')
        new_objective_string = f"{'+'.join([str(value) + f'x{i+1}' for i, value in enumerate(new_objective)])}"
        dual_problem = Simplex(new_objective_string, Objective.MAX.value if self.objective == Objective.MIN.value else Objective.MIN.value)
        
        
        for restriction in new_restriction_strings:
            dual_problem.add_restriction(restriction)
        dual_problem.table = Table.normalize_table(dual_problem.objective_function, dual_problem.table, dual_problem.column_b)
        return dual_problem
    
        

    def show_table(self):
        """Exibe a tabela simplex"""
        table_copy = self.table.copy()
        basic_vars = self.basic_vars.copy()
        table_copy[0] = ["Z"] + table_copy[0]
        for i in range(1, len(table_copy)):
            table_copy[i] = basic_vars[i-1:i ] + [round(value, 3) for value in table_copy[i]]
        print(f'Iteração {self.iterations}')
        print(tabulate(table_copy, tablefmt="fancy_grid", headers=["Base"] + [*self.get_all_vars()] + ["b"]))
    
    def get_bounds(self, x1, x2, limit):
        """Calcula os limites das restrições
        Args:
            x1 (int): Coeficiente da variável x1.
            x2 (int): Coeficiente da variável x2.
            limit (int): Limite da restrição.
        Retorna:
            tuple: Limites das restrições.
        """
        x_1 = [limit/x1, 0.0] if x1 != 0 else [0.0, limit/x2]
        x_2 = [0.0, limit/x2] if x2 != 0 else [limit/x1, 0.0]
        return x_1, x_2
    
    def two_phase(self, verbose= True):
        """Resolve o problema de programação linear utilizando o método das duas fases"""
        num_variaveis =  len(self.expression_util.get_variables(self.string_objective_function)) + self.inserted
        # 1a fase
        
        new_objective = [0] * num_variaveis
        for i in range(num_variaveis):
            if f'x{i+1}' in self.artificial_vars:
                new_objective[i] = -1
            else:
                new_objective[i] = 0
        self.objective_function = new_objective + [0]
        
        self.show_table()
        
        # Monta a funcao artificial
        for var in self.artificial_vars:
            index = self.basic_vars.index(var)
            row = self.table[index+1]
            if row[self.get_all_vars().index(var)] != 1:
                continue
            print(f'Variável artificial: {var} - {row}')
            for i in range(len(row)):
                if f'x{i+1}' in self.artificial_vars:
                    self.objective_function[i] = 0
                else:
                    self.objective_function[i] -= row[i]
                    
        self.table[0] = self.objective_function
        self.show_table()
        original_objective = self.objective
        self.objective = Objective.MIN.value
        self.first_phase = True
        solucao = self.solve()
        to_remove = []
        
        # Remove variáveis artificiais
        for i in range(len(row)):
            if f'x{i+1}' in self.artificial_vars:
                to_remove.append(i)
                self.inserted -= 1
        for i in range(len(self.table)):
            self.table[i] = [value for j, value in enumerate(self.table[i]) if j not in to_remove]
        self.artificial_vars = []
        
        # 2a fase
        self.first_phase = False
        print(f'Objetivo: {self.string_objective_function}')
        self.objective_function = self._build_objective_function(self.string_objective_function)
        
        # calcula a nova função objetivo
        for i in range(self.inserted):
            self.objective_function.append(0)
        cj = self.objective_function + [0]
        cb = []
        for var in self.basic_vars:
            try:
                index = self.get_all_vars().index(var)
                cb.append(cj[index])
            except:
                cb.append(0)
        A = self.table[1:]
        new_objective_function = np.transpose(cj) - np.dot(cb,A)
        self.objective = original_objective
        self.table[0] = [value for value in new_objective_function]
        
        operations = []
        operations.append([f'W = {self.objective_function}'] + [f'Z = {self.table[0]}'])
        try:
            solucao['iteracoes'].append(self.register_iteration(self.pivot_column_index, self.get_pivot_line(self.pivot_column_index), operations))
        except:
            pass
        self.solve()
        solucao['solucao'] = self.get_results()
        self.iteration_history.append(self.register_iteration(None, None, None))
        solucao['iteracoes'] = self.iteration_history 
        solucao['variaveis_nao_basicas'] = {value:0 for key, value in enumerate(self.get_all_vars()) if value not in self.basic_vars}
        solucao['solucao_inteira'] = all(round(value, 3).is_integer() for key, value in solucao['solucao'].items() if key in self.expression_util.get_variables(self.string_objective_function))
        return solucao
    
    
def graphical_method(simplex):
    """Resolve o problema de programação linear utilizando o método gráfico
    Args:
        simplex (Simplex): Instância da classe Simplex.
    Retorna:
        figname (str): Nome do arquivo da imagem gerada.
    Raises:
        Exception: Caso o problema não possua duas variáveis.
    
    Calcula os limites das restrições e plota o gráfico do problema.
    """
    if simplex.expression_util.get_variables(simplex.string_objective_function) != ['x1', 'x2']:
        raise Exception("O método gráfico só pode ser utilizado para problemas com duas variáveis")
    constraints = simplex.restriction_strings
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    constraints = [simplex.get_constraint_from_string(constraint) for constraint in constraints]
    bounds_x = []
    bounds_y = []
    constraint_formulas = []
    
    # calcula os limites das restrições
    for constraint in constraints:
        x, y = simplex.get_bounds(constraint[0][0], constraint[0][1], float(constraint[2][1]))
        bounds_x.append(x)
        bounds_y.append(y)
        constraint_formulas.append(constraint[0][0] * x1 + constraint[0][1] * x2 - float(constraint[2][1]))
        
    # calcula a solução ótima e plota o gráfico
    optimal_solution = float('inf') if simplex.objective == Objective.MIN.value else float('-inf')
    objective_formula = simplex.objective_function[0] * x1 + simplex.objective_function[1] * x2
    for i in range(len(constraint_formulas)):
        for j in range(i + 1, len(constraint_formulas)):
            solution = solve((constraint_formulas[i], constraint_formulas[j]), (x1, x2), dict=True)
            
            if not all(constraint_formula.subs({x1: solution[0][x1], x2: solution[0][x2]}) <= 0 for constraint_formula in constraint_formulas):
                continue
            
            if simplex.objective == Objective.MAX.value:
                if objective_formula.subs({x1: solution[0][x1], x2: solution[0][x2]}) > optimal_solution:
                    optimal_solution = objective_formula.subs({x1: solution[0][x1], x2: solution[0][x2]})
                    optimal_solution_values = solution[0]
            else:
                if objective_formula.subs({x1: solution[0][x1], x2: solution[0][x2]}) < optimal_solution:
                    optimal_solution = objective_formula.subs({x1: solution[0][x1], x2: solution[0][x2]})
                    optimal_solution_values = solution[0]
            
    x_1, x_2 = optimal_solution_values[x1], optimal_solution_values[x2]
    plt.plot(x_1, x_2, 'bo', label=f'Ponto Ótimo')
    
    # plota as restrições
    for i in range(len(constraints)):
        x, y = bounds_x[i], bounds_y[i]
        plt.plot(x, y, label=f"{constraints[i][0][0]}x1 + {constraints[i][0][1]}x2 {constraints[i][1]} {constraints[i][2][1]}")
        
    feasible_region_points = {f'{constraints[i][0][0]}x1 + {constraints[i][0][1]}x2 {constraints[i][1]} {constraints[i][2][1]}': [] for i in range(len(constraints))}
    np_constraints = np.array([constraint[0] + [float(constraint[2][1])] for constraint in constraints])
    x_array = np.linspace(0, 10, 100, dtype=float)

    for x1 in x_array:
        for x2 in x_array:
            for index, constraint in enumerate(np_constraints):
                if constraints[index][1] == "<=":
                    if constraint[0] * x1 + constraint[1] * x2 <= constraint[2]:
                        feasible_region_points[f'{constraints[index][0][0]}x1 + {constraints[index][0][1]}x2 {constraints[index][1]} {constraints[index][2][1]}'].append([x1, x2])
                elif constraints[index][1] == ">=":
                    if constraint[0] * x1 + constraint[1] * x2 >= constraint[2]:
                        feasible_region_points[f'{constraints[index][0][0]}x1 + {constraints[index][0][1]}x2 {constraints[index][1]} {constraints[index][2][1]}'].append([x1, x2])
                elif constraints[index][1] == "=":
                    if constraint[0] * x1 + constraint[1] * x2 == constraint[2]:
                        feasible_region_points[f'{constraints[index][0][0]}x1 + {constraints[index][0][1]}x2 {constraints[index][1]} {constraints[index][2][1]}'].append([x1, x2])

    common_points = []
    for key, value in feasible_region_points.items():
        values_set = set([tuple(point) for point in value])
        common_points = values_set if not common_points else common_points.intersection(values_set)
    common_points = np.array(list(common_points))
    plt.scatter(common_points[:, 0], common_points[:, 1], color='lightgreen', label='Região viável')
    
    # plot the objective function
    x = np.linspace(0, 10, 100)
    y = (optimal_solution - simplex.objective_function[0] * x) / simplex.objective_function[1]
    plt.plot(x, y, label=f"{simplex.objective_function[0]}x1 + {simplex.objective_function[1]}x2 = {round(optimal_solution, 3)}", color='red', linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.xlim(0, max(bounds_x[0]) + 1)
    plt.ylim(0, max(bounds_y[0]) + 1)
    plt.title(f'Solução Gráfica - {"Max" if simplex.objective == Objective.MAX.value else "Min"} Z = {simplex.string_objective_function}')
    plt.legend()
    
    figname = f'static/plots/graphical_solution_{simplex.string_objective_function}.png'
    plt.savefig(figname)
    
    plt.show(block=False)
    
    return figname
    

if __name__ == "__main__":
    objective = "400x1 + 600x2 + 900x3 + 350x4"
    constraints = ["200x2 + 300x3 <= 14000",
                   "20000x2+400000x3+50000x4 <= 12750000",
                   "1x1 + 1x2 + 1x3 + 1x4 = 100",
                   "x4 >= 20",
                   ]
    
    objective = "3x1 + 2x2"
    constraints = ["1x1 + 1x2 <= 4",
                   "x1 <= 2",
                   "1x2 <= 3",
                   ]
    
    
    objective = "5x1 + 4x2 + 3x3"
    constraints = ["2x1 + 3x2 + x3 >= 10",
                   "4x1 + x2 + 2x3 >= 8",
                   "3x1 + 4x2 + 2x3 >= 12",
                   ]
    
    objective = "0.4x1 + 0.5x2"
    constraints = ["0.3x1 + 0.1x2 <= 2.7",
                     "0.5x1 + 0.5x2 <= 6",
                     "0.6x1 + 0.4x2 >= 6",
                     ]

    simplex = Simplex(objective, Objective.MIN.value)
    for constraint in constraints:
        simplex.add_restriction(constraint)
    simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
    print(simplex.dual())
    simplex.solve()

