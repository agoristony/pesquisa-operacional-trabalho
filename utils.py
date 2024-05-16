from typing import Any, Dict, List
import re

class Table:
    @classmethod
    def _get_solution(cls, table: List[List[int]]) -> int:
        for row in table:
            if row[0] == 1:
                return row[-1]
        return 0

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
        basic_variables.remove(0)
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