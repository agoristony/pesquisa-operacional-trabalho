from typing import Any, Dict, List
import re
from sympy import O, Symbol, solve
import matplotlib.pyplot as plt
import numpy as np

class Table:
    @classmethod
    def _get_solution(cls, table: List[List[int]]) -> int:
        return table[0][-1]

    @classmethod
    def _get_basic_vars(cls, table: List[List[int]]) -> list:
        basics = []
        for i in range(len(table[0])):
            basic = 0
            for j in range(len(table)):
                basic += abs(table[j][i])

            if basic == 1:
                basics.append(i)

        return basics

    @classmethod
    def normalize_table(cls, objective_function, table: List[List[int]], column_b: List[int]):
        """Configura as variáveis para cada linha na tabela"""
        table.insert(0, objective_function) 
        normal_size = len(objective_function)
        for row in table:
            if len(row) < normal_size:
                addition = normal_size - len(row)
                for _ in range(addition):
                    row.append(0)
        return [x + [y] for x, y in zip(table, column_b)]

    @classmethod
    def get_results(
        cls, table: List[List[int]], variables: List[str]
    ) -> Dict[str, Any]:
        basic_variables = cls._get_basic_vars(table)

        result = {
            "solucao": cls._get_solution(table),
        }
        try:
            for index in basic_variables:
                variable = variables[index - 1]
                for j in range(len(table)):
                    value = table[j][index]
                    if value == 1:
                        result[variable] = table[j][-1]
                        break
        except Exception:
            pass

        for variable in variables:
            if variable not in result:
                result[variable] = 0

        return result
    
class ExpressionUtil:
    def __sanitize_expression(self, expression: str):
        return expression.replace(" ", "")

    def __is_duplicated(self, collection) -> bool:
        return len(collection) != len(set(collection))
    
    def __get_coefficient(self, monomial: str) -> str:
        if value := re.findall(r'^([+-]?\d*\.?\d+)', monomial):
            return value[0]
        return "1"
    
    def get_variables(self, expression: str):
        pattern = "[a-z][0-9]+"
        return re.findall(pattern, expression)
    
    def __validate_variables(self, expression: str) -> None:
        variables = self.get_variables(expression)
        if self.__is_duplicated(variables):
            raise TypeError("Existe incógnitas repetidas na expressão informada")

        if variables != sorted(variables):
            raise ValueError("Utilize incógnitas em sequência ordenada!")
    
    def __get_algebraic_expressions(self, expression: str):
        # pattern = ">=|\\+|\\-|<="
        negative_pattern = "([+-])"
        splitted = re.split(negative_pattern, expression)
        for i in range(len(splitted)):
            if splitted[i] == "-":
                splitted[i+1] = splitted[i] + splitted[i+1]
                splitted[i] = ""
            elif splitted[i] == "+":
                splitted[i] = ""
            
        splitted = list(filter(None, splitted))
        return splitted
    
    def get_numeric_values(self, expression: str, fo_variables: list):
        expression = self.__sanitize_expression(expression)

        self.__validate_variables(expression)

        algebraic_expressions = self.__get_algebraic_expressions(expression)
        values = {variable: 0 for variable in fo_variables}
        for variable in fo_variables:
            for monomial in algebraic_expressions:
                if variable in monomial:
                    value = self.__get_coefficient(monomial)
                    values[variable] = value
                    break           
        return [float(value) for value in values.values()]
    
def get_bound(x1, x2, limit):
    x_1 = [limit/x1, 0.0] if x1 != 0 else [0.0, limit/x2]
    x_2 = [0.0, limit/x2] if x2 != 0 else [limit/x1, 0.0]
    return x_1, x_2

def crossing_point(constraint1, constraint2):
    x1, y1, limit1 = constraint1
    x2, y2, limit2 = constraint2
    x = (limit1 * y2 - limit2 * y1) / (x1 * y2 - x2 * y1)
    y = (limit1 * x2 - limit2 * x1) / (y1 * x2 - y2 * x1)
    return x, y

def primal_to_dual(problem):
    """
    Converte um problema de programação linear primal em um problema de programação linear dual.
    """
    dual_problem = problem.copy()
    dual_problem.objective_function = [1] + problem.column_b
    dual_problem.column_b = problem.objective_function[1:]
    dual_problem.table = [[1] + row for row in problem.table]
    dual_problem.inserted = len(problem.table[0])
    return dual_problem
    
def graphical_solution(problem):
    s1 = Symbol("x1")
    s2 = Symbol("x2")
    print(f'Coluna B: {problem.column_b}')
    constraints = [constraint[1:problem.inserted] + [problem.column_b[i]] for i, constraint in enumerate(problem.table)]
    # constraints = [(constraint[0], constraint[1], constraint[2]) for constraint in constraints]
    print(f"Constraints: {constraints}")
    objective = [num * -1 for num in problem.objective_function[1:3]]

    # Inicializa listas para armazenar os limites das restriçõe.
    bounds_x = []
    bounds_y = []
    constraint_formulas = []

    # Calcula os limites das restriçõe.
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

    # Filtra os pontos que estão dentro da região viável removendo os que não satisfazem as restrições
    feasible_region_points = [
        (x, y) for x, y in feasible_region_points
        if x >= 0 and y >= 0 and all(constraint.subs({s1: x, s2: y}) <= 0 for constraint in constraint_formulas)
    ]

    # achar o ponto ótimo
    x1_profit, x2_profit = objective
    max_profit_formula = x1_profit * s1 + x2_profit * s2
    optimal_point = max(feasible_region_points, key=lambda point: max_profit_formula.subs({'x1': point[0], 'x2': point[1]}))

    max_profit = float(max_profit_formula.subs({'x1': optimal_point[0], 'x2': optimal_point[1]}))
    
    # plotar os pontos de intersecção
    # points = [crossing_point(constraint1, constraint2) for constraint1 in constraints for constraint2 in constraints if constraint1 != constraint2]
    # for point in points:
    #     if point[0] >= 0 and point[1] >= 0:
    #         plt.plot(*point, marker='o', color='black')

    # plotar as restrições
    colors = ['darkgreen', 'darkblue', 'darkred', 'darkorange', 'purple']

    for i, (x, y, constraint) in enumerate(zip(bounds_x, bounds_y, constraint_formulas)):
        plt.plot(x, y, linestyle=':', marker='o', color=colors[i % len(colors)])
        
        plt.annotate(
            constraint,
            (x[1], y[1])
        )

    # plota a região viável e as restricoes de não negatividade
    plt.plot(0, 0, marker='o')
    plt.plot([0, 0], [max([b[1] for b in bounds_y]), 0], linestyle=':', color='grey')
    plt.plot([max([b[0] for b in bounds_x]), 0], [0, 0], linestyle=':', color='grey')

    feasible_region_points = sorted(feasible_region_points, key=lambda x: x[0])
    feasible_region_points.append((0, 0))
    x, y = zip(*feasible_region_points)
    plt.fill(x, y, 'lightgrey', alpha=0.5)

    # plota o ponto ótimo
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='red')

    # plota a curva de nível
    x = np.linspace(0, max([b[0] for b in bounds_x]), 100)
    y = (max_profit - objective[0] * x) / objective[1]
    plt.plot(x, y, label=f"{objective[0]}x1 + {objective[1]}x2 = {max_profit}", color='red')
    
    # imprime os pontos viáveis
    points = ["O"] + [f"A{i}" for i in range(1, len(feasible_region_points))]
    for point, (x, y) in zip(points, feasible_region_points):
        print(f"{point} = ({x}, {y})")

    plt.grid()
    plt.xlabel(f"x1", color='darkblue')
    plt.ylabel(f"x2", color='darkgreen')
    plt.show()