import numpy as np
from numpy import linalg as LA
from tabulate import tabulate
from numpy.linalg import inv

class Simplex:
    def __init__(self):
        self.table = []
        self.iterations = 0

    def set_objective_function(self, c: list):
        self.table.append(c)

    def add_restrictions(self, sa: list):
        self.table.append(sa)
    
    def get_entry_column(self):
        pivot_column = min(self.table[0])
        index = self.table[0].index(pivot_column)
        return index
    
    def get_exit_row(self, entry_column: int):
        results = {}
        for line in range(1, len(self.table)):
            if self.table[line][entry_column] > 0:
                results[line] = self.table[line][-1] / self.table[line][entry_column]
        return min(results, key=results.get)
    
    def calculate_pivot_line(self, exit_line: int, entry_column: int) -> list:
        pivot = self.table[exit_line][entry_column]
        return [value / pivot for value in self.table[exit_line]]

    def calculate_new_line(self, line: list, entry_column: int, pivot_line: list):
        pivot = line[entry_column] * -1
        result_line = [value * pivot for value in pivot_line]
        return [sum(x) for x in zip(line, result_line)]
    
    def is_negative(self) -> bool:
        return min(self.table[0]) < 0
    
    def show_table(self):
        print(f'Iteração: {self.iterations}')
        print(tabulate(self.table, tablefmt="fancy_grid", headers=["Z"] + [f"x{i}" for i in range(1, len(self.table[0]) - 1)] + ["b"]))
    
    def calculate(self):
        entry_column = self.get_entry_column()
        exit_row = self.get_exit_row(entry_column)
        pivot_line = self.calculate_pivot_line(exit_row, entry_column)
        self.table[exit_row] = pivot_line
        table_copy = self.table.copy()
        for line in range(len(self.table)):
            if line != exit_row:
                self.table[line] = self.calculate_new_line(table_copy[line], entry_column, pivot_line)
        self.iterations += 1

    def solve(self):
        self.calculate()
        while self.is_negative():
            self.calculate()
            self.show_table()
        self.show_table()
        print(f'Valor ótimo: {self.table[0][-1]}')
        print('Variáveis básicas:')
        for i in range(1, len(self.table[0]) - 1):
            for j in range(1, len(self.table)):
                if self.table[j][i] == 1:
                    print(f'x{i} = {self.table[j][-1]}')
                    break


if __name__ == "__main__":
    simplex = Simplex()
    simplex.set_objective_function([1,-3, -2, 0, 0, 0 ,0])
    simplex.add_restrictions([0,2, 1, 1, 0, 0, 18])
    simplex.add_restrictions([0,2, 3, 0, 1, 0, 42])
    simplex.add_restrictions([0,3, 1, 0, 0, 1, 24])
    simplex.solve()
