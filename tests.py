import unittest
from simplex import Simplex, Objective
from utils import Table

class TestSimplex(unittest.TestCase):

    def test_initialization(self):
        objective = "3x1 + 2x2"
        simplex = Simplex(objective, Objective.MAX.value)
        self.assertEqual(simplex.string_objective_function, objective)
        self.assertEqual(simplex.objective, Objective.MAX.value)

    def test_add_restriction(self):
        objective = "3x1 + 2x2"
        constraints = ["1x1 + 2x2 <= 10"]
        simplex = Simplex(objective, Objective.MAX.value)
        for constraint in constraints:
            simplex.add_restriction(constraint)
            self.assertIn(10, simplex.column_b)
        self.assertEqual(len(simplex.table), 1)
        self.assertEqual(simplex.table[0][0], 1)  # Coeficiente de x1
        self.assertEqual(simplex.table[0][1], 2)  # Coeficiente de x2
        

    def test_solve_simple_problem(self):
        objective = "5x1 + 6x2"
        constraints = ["x1 + x2 <= 5",
                   "4x1 + 7x2 <= 28",
                   ]
        simplex = Simplex(objective, Objective.MAX.value)
        for constraint in constraints:
            restriction, column_b = simplex.add_restriction(constraint)
            self.assertEqual(float(column_b), simplex.column_b[-1])
        simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        self.assertIn('x1', solution)
        self.assertIn('x2', solution)
        self.assertEquals(solution['solucao'], 2.333333333333333)

    def test_solve_another_problem(self):
        objective = "5x1 + 4x2"
        constraints = ["3x1 + 2x2 >= 5",
                       "2x1 + 3x2 >= 7"]
        simplex = Simplex(objective, Objective.MAX.value)
        for constraint in constraints:
            simplex.add_restriction(constraint)
        simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        self.assertIn('x1', solution)
        self.assertIn('x2', solution)

    # def test_two_phase_method(self):
    #     objective = "400x1 + 600x2 + 900x3 + 350x4"
    #     constraints = ["200x2 + 300x3 <= 14000",
    #                    "20000x2+400000x3+50000x4 <= 12750000",
    #                 #    "1x1 + 1x2 + 1x3 + 1x4 = 100",
    #                    "x4 >= 20"]
    #     simplex = Simplex(objective, Objective.MAX.value)
    #     for constraint in constraints:
    #         simplex.add_restriction(constraint)
    #     simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
    #     solution = simplex.solve()
    #     self.assertIsNotNone(solution)
    #     self.assertIn('x1', solution)
    #     self.assertIn('x2', solution)
    #     self.assertIn('x3', solution)
    #     self.assertIn('x4', solution)

if __name__ == '__main__':
    unittest.main()
