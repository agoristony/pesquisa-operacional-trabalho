
from tabulate import tabulate
from utils import Table, ExpressionUtil
from fractions import Fraction

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
        row = [coef * (-1) for coef in self.expression_util.get_numeric_values(string_objective_function)]
        return [1] + row
    

    def add_restriction(self, expression: str):
        delimiter = "<="
        default_format = True

        # if not self.is_simplex_standard(expression):
        #     raise ValueError("Simplex Duas Fases não implementado!")

        splitted_expression = expression.split(delimiter)
        constraint = [0] + self.expression_util.get_numeric_values(
            splitted_expression[0]
        )

        if not default_format:
            self.objective_function += [0]
        if delimiter == "<=":
            constraint = self.insert_slack_var(constraint, default_format)
        self.column_b.append(int(splitted_expression[1]))
        self.table.append(constraint)
        
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
        column = self.get_entry_column()
        # linha que vai sair
        first_exit_line = self.get_pivot_line(column)

        line = table[first_exit_line]
        # identificando o pivo da linha que vai sair
        pivot = line[self.pivot_column_index]
        self.entering_vars.append(f"x{self.pivot_column_index}")
        self.leaving_vars.append(self.basic_vars[first_exit_line - 1])
        print(f'A variavel que sai da base é a {self.entering_vars[self.iterations - 1]} e a que entra é {self.leaving_vars[self.iterations - 1]}')


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
        

if __name__ == "__main__":
    simplex = Simplex("3x1 + 2x2")
    simplex.add_restriction("2x1 + 1x2 <= 18")
    simplex.add_restriction("2x1 + 3x2 <= 42")
    simplex.add_restriction("3x1 + 1x2 <= 24")
    print(simplex.solve())

