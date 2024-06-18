import math
from simplex import Simplex, Objective, Table

class BranchAndBound:
    def __init__(self, objective_function, constraints, objective=Objective.MAX.value):
        self.objective_function = objective_function
        self.constraints = constraints
        self.objective = objective
        self.best_solution = None
        self.best_objective_value = -math.inf if objective == Objective.MAX.value else math.inf
        self.history = []

    def solve(self):
        initial_simplex = Simplex(self.objective_function, self.objective)
        
        for constraint in self.constraints:
            initial_simplex.add_restriction(constraint)
        initial_simplex.table = Table.normalize_table(initial_simplex.objective_function, initial_simplex.table, initial_simplex.column_b)
        sol = initial_simplex.solve(True)
        self.history.append({'simplex': initial_simplex, 'solution': sol, 'integer_solution': False})
        self.best_solution, self.best_objective_value = [round(sol['solucao'][f'{var}'],6) for var in initial_simplex.basic_vars], sol['solucao']['solucao']
        
        objective_value, variable_values = sol['solucao']['solucao'], {var:round(sol['solucao'][f'{var}'],6) for var in initial_simplex.basic_vars if var in initial_simplex.expression_util.get_variables(initial_simplex.string_objective_function)}
        integer_solution, fractional_var_index = self._check_integer_solution(variable_values)
        if integer_solution:
            print(f'Solucao inteira: {variable_values} - {objective_value}')
            self.best_solution, self.best_objective_value = variable_values, objective_value
            self.history[-1]['integer_solution'] = True
            return self.history
        lower_bound_simplex, upper_bound_simplex = self._create_subproblems(initial_simplex, fractional_var_index, variable_values[fractional_var_index])
        self.history[-1]['lower_bound_simplex'] = lower_bound_simplex.restriction_strings[-1]
        self.history[-1]['upper_bound_simplex'] = upper_bound_simplex.restriction_strings[-1]
        solutions = [self._branch_and_bound(lower_bound_simplex), self._branch_and_bound(upper_bound_simplex)]
        if solutions[0] is None and solutions[1] is None:
            return None
        return self.history

    def _branch_and_bound(self, simplex):
        solution = simplex.solve(False)
        self.history.append({'simplex': simplex, 'solution': solution, 'integer_solution': False})
        if solution is None:
            return
        objective_value, variable_values = solution['solucao']['solucao'], {var:round(solution['solucao'][f'{var}'],6) for var in simplex.basic_vars if var in simplex.expression_util.get_variables(simplex.string_objective_function)}
        integer_solution, fractional_var_index = self._check_integer_solution(variable_values)
        if integer_solution:
            print(f'Solucao inteira: {variable_values} - {objective_value}')
            self.best_solution, self.best_objective_value = variable_values, objective_value
            self.history[-1]['integer_solution'] = True
            return self.history
        if self.objective == Objective.MIN.value:
            if objective_value <= self.best_objective_value:
                return
        else:
            if objective_value > self.best_objective_value:
                return
        print(f'criando subproblemas - {fractional_var_index} - {variable_values}')
        lower_bound_simplex, upper_bound_simplex = self._create_subproblems(simplex, fractional_var_index, variable_values[fractional_var_index])
        self.history[-1]['lower_bound_simplex'] = lower_bound_simplex.restriction_strings[-1]
        self.history[-1]['upper_bound_simplex'] = upper_bound_simplex.restriction_strings[-1]
        
        try:
            self._branch_and_bound(lower_bound_simplex)
        except:
            print(f'Erro ao resolver o problema inferior - {lower_bound_simplex.restriction_strings[-1]}')
        
        try:
            self._branch_and_bound(upper_bound_simplex)
        except:
            print(f'Erro ao resolver o problema superior - {upper_bound_simplex.restriction_strings[-1]}')

    def _check_integer_solution(self, variable_values):          
        decimal_parts = {key: math.floor(value) for key, value in variable_values.items()}
        for key, value in variable_values.items():
            if not value.is_integer():
                return False, max(decimal_parts, key=decimal_parts.get)
        return True, None

    def _create_subproblems(self, simplex, var_index, fractional_value):
        
        lower_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)
        upper_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)

        lower_constraint = f"{var_index} <= {math.floor(fractional_value)}"
        upper_constraint = f"{var_index} >= {math.ceil(fractional_value)}"
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
    objective = "220x1 + 80x2"
    constraints = ["5x1 + 2x2 <= 16",
                   "2x1 - 1x2 <= 4",
                   "-1x1 + 2x2 <= 4",
                   ]
    objective = "0.4x1 + 0.5x2"
    constraints = ["0.3x1 + 0.1x2 <= 2.7",
                   "0.5x1 + 0.5x2 <= 6",
                   "0.6x1 + 0.4x2 >= 6",
                   ]
    objective = "5x1 + 4x2"
    constraints = ["3x1 + 2x2 >= 5",
                   "2x1 + 3x2 >= 7",
                   ]
    bnb = BranchAndBound(objective, constraints, Objective.MIN.value)
    print(f'Melhor solucao: {bnb.solve()}')

