import math
from simplex import Simplex, Objective, Table

def mathjax_string(string):
    # string = "5x1 + 2x2 <= 16"
    # \(5x_1 + 2x_2 \leq 16\)
    new_string = string.replace('<=', r'\leq').replace('>=', r'\geq').replace('=', r'=').replace('x', r'x_')
    return f'\[{new_string}\]'
    

class BranchAndBound:
    def __init__(self, objective_function, constraints, objective=Objective.MAX.value):
        self.objective_function = objective_function
        self.constraints = constraints
        self.objective = objective
        self.best_solution = None
        self.best_objective_value = -math.inf if objective == Objective.MAX.value else math.inf
        self.best_integer_solution = None
        self.history = []

    def solve(self, verbose=False):
        initial_simplex = Simplex(self.objective_function, self.objective)
        
        for constraint in self.constraints:
            initial_simplex.add_restriction(constraint)
        initial_simplex.table = Table.normalize_table(initial_simplex.objective_function, initial_simplex.table, initial_simplex.column_b)
        sol = initial_simplex.solve(True)
        id = len(self.history)
        self.history.append({'simplex': initial_simplex, 'solution': sol, 'integer_solution': False, 'id': id, 'parent_id': None, 'level': 0})
        self.best_solution, self.best_objective_value = [round(sol['solucao'][f'{var}'],6) for var in initial_simplex.basic_vars], sol['solucao']['solucao']
        
        objective_value, variable_values = sol['solucao']['solucao'], {var:round(sol['solucao'][f'{var}'],6) for var in initial_simplex.basic_vars if var in initial_simplex.expression_util.get_variables(initial_simplex.string_objective_function)}
        integer_solution, fractional_var_index = self._check_integer_solution(variable_values)
        if integer_solution:
            print(f'Solucao inteira: {variable_values} - {objective_value}')
            self.best_solution, self.best_objective_value = variable_values, objective_value
            self.history[-1]['integer_solution'] = True
            return self.history, self.best_solution, self.best_objective_value
        lower_bound_simplex, upper_bound_simplex = self._create_subproblems(initial_simplex, fractional_var_index, variable_values[fractional_var_index])
        self.history[-1]['lower_bound_simplex'] = lower_bound_simplex.restriction_strings[-1]
        self.history[-1]['upper_bound_simplex'] = upper_bound_simplex.restriction_strings[-1]
        solutions = [self._branch_and_bound(lower_bound_simplex, id), self._branch_and_bound(upper_bound_simplex, id)]
        if solutions[0] is None and solutions[1] is None:
            return None
        return self.history, self.best_solution, self.best_objective_value

    def _branch_and_bound(self, simplex, parent_id=None):
        solution = simplex.solve(False)
        id = len(self.history)
        if solution is None:
            self.history.append({'simplex': simplex, 'solution': {'solucao': {'solucao': '-'}}, 'integer_solution': False, 'id': len(self.history), 'parent_id': id-2, 'level': self.history[id-1]['level']})
            return self.history
        solution['solucao'] = {key: round(value, 6) for key, value in solution['solucao'].items()  if key in simplex.expression_util.get_variables(simplex.string_objective_function) or key == 'solucao'}
        self.history.append({'simplex': simplex, 'solution': solution, 'integer_solution': False, 'id': id, 'parent_id': parent_id, 'level': self.history[parent_id]['level'] + 1})
        
        objective_value, variable_values = solution['solucao']['solucao'], {var:round(solution['solucao'][f'{var}'],6) for var in simplex.basic_vars if var in simplex.expression_util.get_variables(simplex.string_objective_function)}
        integer_solution, fractional_var_index = self._check_integer_solution(variable_values)
        if integer_solution:
            print(f'Solucao inteira: {variable_values} - {objective_value}')
            self.best_solution, self.best_objective_value = variable_values, objective_value
            self.history[-1]['integer_solution'] = True
            return self.history
        # if self.objective == Objective.MIN.value:
        #     if objective_value <= self.best_objective_value:
        #         return
        # else:
        #     if objective_value > self.best_objective_value:
        #         return
        print(f'criando subproblemas - {fractional_var_index} - {variable_values}')
        lower_bound_simplex, upper_bound_simplex = self._create_subproblems(simplex, fractional_var_index, variable_values[fractional_var_index])
        self.history[-1]['lower_bound_simplex'] = lower_bound_simplex.restriction_strings[-1]
        self.history[-1]['upper_bound_simplex'] = upper_bound_simplex.restriction_strings[-1]
        
        try:
            self._branch_and_bound(lower_bound_simplex, id)
        except Exception as e:
            self.history.append({'simplex': lower_bound_simplex, 'solution': {'solucao': {'solucao': '-'}}, 'integer_solution': False, 'id': len(self.history), 'parent_id': id, 'level': self.history[id]['level'] + 1})
            print(f'Erro ao resolver o problema inferior - {lower_bound_simplex.restriction_strings[-1]} - {e}')
        
        try:
            self._branch_and_bound(upper_bound_simplex, id)
        except Exception as e:
            self.history.append({'simplex': upper_bound_simplex, 'solution': {'solucao': {'solucao': '-'}}, 'integer_solution': False, 'id': len(self.history), 'parent_id': id, 'level': self.history[id]['level'] + 1})
            print(f'Erro ao resolver o problema superior - {upper_bound_simplex.restriction_strings[-1]} - {e}')

    def _check_integer_solution(self, variable_values):          
        decimal_parts = {key: value - math.floor(value) for key, value in variable_values.items() if not value.is_integer()}
        for key, value in variable_values.items():
            if not value.is_integer():
                return False, min(decimal_parts, key=decimal_parts.get)
        return True, None

    def _create_subproblems(self, simplex, var_index, fractional_value):
        
        lower_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)
        upper_bound_simplex = Simplex(simplex.string_objective_function, simplex.objective)

        lower_constraint = f"{var_index} <= {math.floor(fractional_value)}"
        upper_constraint = f"{var_index} >= {math.ceil(fractional_value)}"
        # if f"{var_index} >= {math.floor(fractional_value)}" in simplex.restriction_strings:
        #     print('Restricao de igualdade')
        #     lower_constraint = f"{var_index} = {math.floor(fractional_value)}"
        #     simplex.restriction_strings.remove(f"{var_index} >= {math.floor(fractional_value)}")
        # print(f"{var_index} <= {math.ceil(fractional_value)}")
        # if f"{var_index} <= {math.ceil(fractional_value)}" in simplex.restriction_strings:
        #     print('Restricao de igualdade')
        #     upper_constraint = f"{var_index} = {math.ceil(fractional_value)}"
        #     simplex.restriction_strings.remove(f"{var_index} <= {math.ceil(fractional_value)}")
        # print(f'Restricao inferior: {lower_constraint} Restricao superior: {upper_constraint}')

        for constraint in simplex.restriction_strings:
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
    # objective = "0.4x1 + 0.5x2"
    # constraints = ["0.3x1 + 0.1x2 <= 2.7",
    #                "0.5x1 + 0.5x2 <= 6",
    #                "0.6x1 + 0.4x2 >= 6",
    #                ]
    # objective = "5x1 + 4x2"
    # constraints = ["3x1 + 2x2 >= 5",
    #                "2x1 + 3x2 >= 7",
    #                ]
    bnb = BranchAndBound(objective, constraints, Objective.MAX.value)
    print(f'Melhor solucao: {bnb.solve()}')

