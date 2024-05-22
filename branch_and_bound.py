import math
from simplex import Simplex, Objective, Table

class BranchAndBound:
    def __init__(self, objective_function, constraints, objective=Objective.MAX.value):
        self.objective_function = objective_function
        self.constraints = constraints
        self.objective = objective
        self.best_solution = None
        self.best_objective_value = -math.inf if objective == Objective.MAX.value else math.inf

    def solve(self):
        initial_simplex = Simplex(self.objective_function, self.objective)
        for constraint in self.constraints:
            initial_simplex.add_restriction(constraint)
        initial_simplex.table = Table.normalize_table(initial_simplex.objective_function, initial_simplex.table, initial_simplex.column_b)
        self._branch_and_bound(initial_simplex)
        return self.best_solution, self.best_objective_value

    def _branch_and_bound(self, simplex):
        solution = simplex.solve()
        if solution is None:
            return
        
        objective_value, variable_values = solution['solucao'], [float(solution['x1']), float(solution['x2'])]
        if self.objective == Objective.MAX.value:
            if objective_value <= self.best_objective_value:
                return
        else:
            if objective_value >= self.best_objective_value:
                return

        integer_solution, fractional_var_index = self._check_integer_solution(variable_values)

        if integer_solution:
            self.best_solution = variable_values
            self.best_objective_value = objective_value
            print(f'Solucao inteira: {variable_values} - {simplex.table[-1][1:]})')
            return
        
        lower_bound_simplex, upper_bound_simplex = self._create_subproblems(simplex, fractional_var_index, variable_values[fractional_var_index])
        
        self._branch_and_bound(lower_bound_simplex)
        self._branch_and_bound(upper_bound_simplex)

    def _check_integer_solution(self, variable_values):
        decimal_parts = [value - math.floor(value) for value in variable_values]           
        for i, value in enumerate(variable_values):
            if not value.is_integer():
                return False, decimal_parts.index(max(decimal_parts))
        return True, None

    def _create_subproblems(self, simplex, var_index, fractional_value):
        
        lower_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)
        upper_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)

        lower_constraint = f"x{var_index + 1} <= {math.floor(fractional_value)}"
        upper_constraint = f"x{var_index + 1} >= {math.ceil(fractional_value)}"
        print(f'Restricao inferior: {lower_constraint} Restricao superior: {upper_constraint}')

        for constraint in self.constraints:
            lower_bound_simplex.add_restriction(constraint)
            upper_bound_simplex.add_restriction(constraint)
            
        lower_bound_simplex.add_restriction(lower_constraint)
        upper_bound_simplex.add_restriction(upper_constraint)

        lower_bound_simplex.table = Table.normalize_table(lower_bound_simplex.objective_function, lower_bound_simplex.table, lower_bound_simplex.column_b)
        upper_bound_simplex.table = Table.normalize_table(upper_bound_simplex.objective_function, upper_bound_simplex.table, upper_bound_simplex.column_b)

        return lower_bound_simplex, upper_bound_simplex

if __name__ == "__main__":
    # objective = "220x1 + 80x2"
    # constraints = ["5x1 + 2x2 <= 16",
    #                "2x1 - 1x2 <= 4",
    #                "-1x1 + 2x2 <= 4",
    #                ]
    objective = "5x1 + 4x2"
    constraints = ["3x1 + 2x2 >= 5",
                   "2x1 + 3x2 >= 7"
                   ]
    bnb = BranchAndBound(objective, constraints, Objective.MIN.value)
    best_solution, best_value = bnb.solve()
    print(f'Melhor solucao: {best_solution}')
    print(f'Melhor valor: {best_value}')
