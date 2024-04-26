from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/simplex', methods=['POST'])
def simplex():
    data = request.json
    try:
        result = simplex_solver(data)
        response = {
            "status": "Sucesso",
            "resumo": {
                "numeroDeIteracoes": len(result['tableaus']) - 1,
                "valorOtimo": result['optimalValue'],
                "valoresDasVariaveis": result['solution'],
                "variaveisBasicas": result['basicVariables']
            },
            "iteracoes": result['iterations']
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"status": "Falha", "mensagem": str(e)}), 500

def simplex_solver(data):
    num_variaveis = data['num_variaveis']
    objective = data['objective']
    restricoes = data['restricoes']

    c = [-coef for coef in objective]  # Converter para maximização
    A = [r['coeficientes'] for r in restricoes]
    b = [r['lado_direito'] for r in restricoes]

    tableau, num_variables = initialize_tableau(c, A, b, num_variaveis)
    tableaus = [clone_tableau(tableau)]  # Armazenar o tableau inicial
    iterations = []

    while True:
        pivot = find_pivot(tableau)
        if pivot['column'] == -1:
            break  # Se não há coluna de pivô, solução ótima alcançada

        pivot_tableau(tableau, pivot)
        tableaus.append(clone_tableau(tableau))  # Armazenar cada tableau após pivoteamento
        iterations.append({
            "iteracao": len(tableaus) - 1,
            "tableau": tableau_to_list(tableau),
            "pivot": pivot,
            "variaveisBasicas": identify_basic_variables(tableau, num_variables)
        })

    optimal_value = tableau[0][-1]  # Último valor da linha da função objetivo
    solution = extract_solution(tableau, num_variables)
    basic_variables = identify_basic_variables(tableau, num_variables)

    return {
        "optimalValue": optimal_value,
        "solution": solution.tolist(),  # Convert to list
        "tableaus": tableaus,
        "iterations": iterations,
        "basicVariables": basic_variables
    }

def identify_basic_variables(tableau, num_variables):
    basic_variables = []
    num_rows = len(tableau)
    num_cols = len(tableau[0]) - 1

    for j in range(num_variables):
        is_basic = False
        for i in range(1, num_rows):
            if tableau[i][j] == 1 and is_unit_column(tableau, j, i):
                basic_variables.append({"variable": f"x{j + 1}", "row": i, "value": round(tableau[i][num_cols], 2)})
                is_basic = True
                break
        if not is_basic:
            basic_variables.append({"variable": f"x{j + 1}", "row": None, "value": 0})
    return basic_variables

def initialize_tableau(c, A, b, num_variables):
    num_rows = len(A)
    num_cols = len(c) + num_rows + 1
    tableau = np.zeros((num_rows + 1, num_cols))

    # Preencher a função objetivo
    tableau[0, :len(c)] = c
    tableau[0, -1] = 0  # Constante na função objetivo

    # Preencher as restrições
    for i in range(num_rows):
        tableau[i + 1, :len(c)] = A[i]
        tableau[i + 1, len(c) + i] = 1  # Variável de folga
        tableau[i + 1, -1] = b[i]

    return tableau, len(c)

def find_pivot(tableau):
    num_cols = tableau.shape[1]
    pivot_column = -1
    lowest = 0  # Encontrar coluna com o menor custo negativo

    for j in range(num_cols - 1):
        if tableau[0, j] < lowest:
            lowest = tableau[0, j]
            pivot_column = j

    if pivot_column == -1:
        return {"row": -1, "column": -1}

    pivot_row = select_pivot_row(tableau, pivot_column)

    return {"row": pivot_row, "column": pivot_column}

def select_pivot_row(tableau, pivot_column):
    num_cols = tableau.shape[1]
    pivot_row = -1
    min_ratio = np.inf

    for i in range(1, tableau.shape[0]):
        if tableau[i, pivot_column] > 0:
            ratio = tableau[i, -1] / tableau[i, pivot_column]
            if ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i

    if pivot_row == -1:
        raise Exception("O problema é ilimitado.")
    return pivot_row

def pivot_tableau(tableau, pivot):
    num_rows, num_cols = tableau.shape
    pivot_value = tableau[pivot['row'], pivot['column']]

    # Normalizar a linha do pivô
    tableau[pivot['row']] /= pivot_value

    # Zerar as outras linhas na coluna do pivô
    for i in range(num_rows):
        if i != pivot['row']:
            factor = tableau[i, pivot['column']]
            tableau[i] -= factor * tableau[pivot['row']]

def extract_solution(tableau, num_variables):
    solution = np.zeros(num_variables)
    num_rows, num_cols = tableau.shape

    # Iterar sobre cada variável para determinar se ela é básica
    for j in range(num_variables):
        basic_row = -1
        for i in range(1, num_rows):
            if tableau[i, j] == 1:
                # Verifica se a coluna é unitária
                if is_unit_column(tableau, j, i):
                    basic_row = i
                    break

        if basic_row != -1:
            solution[j] = tableau[basic_row, -1]

    return solution

def is_unit_column(tableau, col, basic_row):
    num_rows = tableau.shape[0]
    for i in range(1, num_rows):
        if i != basic_row and tableau[i, col] != 0:
            return False
    return True

def clone_tableau(tableau):
    return np.copy(tableau)

def tableau_to_list(tableau):
    return tableau.tolist()

if __name__ == '__main__':
    app.run(port=3000)
