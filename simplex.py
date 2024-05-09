
from tabulate import tabulate
from utils import Table, ExpressionUtil
from fractions import Fraction


    
class Simplex:
    def __init__(self, literal_objective_function):
        self.table = []
        self.iterations = 0
        self.pivot_column_index = 0
        self.expression_util = ExpressionUtil()
        self.literal_objective_function = literal_objective_function
        self.column_b = [0]
        self.inserted = 0
        self.objective_function = self._build_objective_function(literal_objective_function)
        self.basic_vars = []

    def _build_objective_function(self, objective_function: str):
        row = list(
            map(
                lambda value: value * (-1),
                self.expression_util.convert_in_calculable_expression(objective_function),
            )
        )
        return [1] + row
    
    def set_objective_function(self, c: list):
        self.table.append(c)

    def add_restrictions(self, expression: str):
        delimiter = "<="
        default_format = True

        # if not self.is_simplex_standard(expression):
        #     raise ValueError("Simplex Duas Fases não implementado!")

        splitted_expression = expression.split(delimiter)
        constraint = [0] + self.expression_util.convert_in_calculable_expression(
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
        """
        Calcula a nova linha que será substituída na tabela
        row (list) -> linha que será trocada
        pivot_line (list) -> linha pivô
        """

        pivot = row[self.pivot_column_index] * -1

        result_line = [pivot * value for value in pivot_line]

        new_line = [new_value + old_value for new_value, old_value in zip(result_line, row)]
        self.print_line_operation(row, pivot_line, new_line)
        return new_line

    def calculate(self, table: list) -> None:
        column = self.get_entry_column()
        # linha que vai sair
        first_exit_line = self.get_pivot_line(column)

        line = table[first_exit_line]
        # identificando o pivo da linha que vai sair
        pivot = line[self.pivot_column_index]
        print(f'A variavel que sai da base é a {self.basic_vars[first_exit_line - 1]} e a que entra é x{self.pivot_column_index}')


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
        self.iterations += 1
        

    def solve(self):
        self.table = Table.normalize_table(self.objective_function, self.table, self.column_b)
        self.show_table()
        self.calculate(self.table)
        self.show_table()
        while not self.is_optimal():
            self.calculate(self.table)
            
            self.show_table()
            
        variables = self.expression_util.get_variables(self.literal_objective_function)
        print(f'Resultados: {Table.get_results(self.table, variables)}')
        return Table.get_results(self.table, variables)
    
    def is_optimal(self) -> bool:
        return min(self.table[0]) >= 0
    
    def insert_slack_var(self, row: list, default_format=True):
        """Insere variável de folga na restrição"""
        self.objective_function.append(0)
        variables = len(self.expression_util.get_variables(self.literal_objective_function))

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
    simplex.add_restrictions("2x1 + 1x2 <= 18")
    simplex.add_restrictions("2x1 + 3x2 <= 42")
    simplex.add_restrictions("3x1 + 1x2 <= 24")
    simplex.solve()

