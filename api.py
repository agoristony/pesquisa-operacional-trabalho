import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from simplex import Simplex, Objective, graphical_method
from utils import Table
import html
from branch_and_bound import BranchAndBound

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

@app.get("/simplex", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="simplex.html"
    )
    
@app.post("/simplex_form", response_class=HTMLResponse)
async def read_item(request: Request):
    form = await request.form()
    variaveis = form['variaveis']
    return templates.TemplateResponse(
        request=request, name="form_simplex.html", context={"variaveis": variaveis,}
    )
    
@app.post("/simplex_solver", response_class=HTMLResponse)
async def read_item(request: Request):
    form = await request.form()
    keys = form.keys()
    tipo_problema = form['tipoProblema']
    tipo_simplex = form['tipoSimplex']
    
    num_vars = len([key for key in keys if key.startswith('a1')])
    num_constraints = len([key for key in keys if key.startswith('b')])
    constraint_types = [form[f'relacao{i}'] for i in range(1, num_constraints + 1)]
    b = [float(form[f'b{i}']) if form[f'b{i}'] != '' else 0 for i in range(1, num_constraints + 1)]
    A = [[float(form[f'a{i}{j}']) if form[f'a{i}{j}'] != '' else 0 for j in range(1, num_vars + 1)] for i in range(1, len(b) + 1)]
    objective_function = [form[f'c{i}'] if form[f'c{i}'] != '' else 0 for i in range(1, num_vars + 1)]
    objective_function_string = '+'.join(objective_function[i-1] + f'x{i}' for i in range(1, num_vars + 1))
    simplex = Simplex(objective_function_string, Objective.MIN.value if tipo_problema == 'min' else Objective.MAX.value)
    for i, constraint in enumerate(A):
        constraint_string = '+'.join(str(constraint[j]) + f'x{j+1}' for j in range(len(constraint)))
        print(constraint_string + constraint_types[i] + str(b[i]))
        simplex.add_restriction(constraint_string + constraint_types[i] + str(b[i]))
    simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
    if 'grafico' in tipo_simplex:
        grafico_inteiro = 'Inteiro' in tipo_simplex
        path, results = graphical_method(simplex, grafico_inteiro)
        print(results)
        solucao_inteira = results['solucao_inteira']
        return templates.TemplateResponse(
            request=request, name="graphical_solver.html", context={"image": path, "inteiro": solucao_inteira}
        )
    if tipo_simplex == 'dual':
        simplex = simplex.dual()
    try:
        solution = simplex.solve(False)
    except:
        return templates.TemplateResponse(
            request=request, name="infeasible_solution.html")
        
    return templates.TemplateResponse(
        request=request, name="simplex_solver.html", context={"solution": solution, "problema": simplex}
    )
    

@app.get("/branch_and_bound", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="branch_and_bound.html"
    )
    
@app.post("/branch_and_bound_form", response_class=HTMLResponse)
async def read_item(request: Request):
    form = await request.form()
    variaveis = form['variaveis']
    return templates.TemplateResponse(
        request=request, name="form_branch_and_bound.html", context={"variaveis": variaveis}
    )
    
def get_group(solution, best_value, i):
        if solution[i]['integer_solution'] and solution[i]['solution']['solucao']['solucao'] == best_value:
            return "best", 'green'
        elif solution[i]['integer_solution']:
            return "integer", 'blue'
        elif solution[i]['solution']['solucao']['solucao'] == '-':
            return "wrong", 'red'
        return "normal", 'orange'
    
@app.post("/branch_and_bound_solver", response_class=HTMLResponse)
async def read_item(request: Request):
    form = await request.form()
    keys = form.keys()
    tipo_problema = form['tipoProblema']
    num_vars = len([key for key in keys if key.startswith('a1')])
    num_constraints = len([key for key in keys if key.startswith('b')])
    constraint_types = [form[f'relacao{i}'] for i in range(1, num_constraints + 1)]
    b = [float(form[f'b{i}']) for i in range(1, num_constraints + 1)]
    A = [[float(form[f'a{i}{j}']) for j in range(1, num_vars + 1)] for i in range(1, len(b) + 1)]
    objective_function = [form[f'c{i}'] for i in range(1, num_vars + 1)]
    objective_function_string = '+'.join(objective_function[i-1] + f'x{i}' for i in range(1, num_vars + 1))
    constraints = []
    for i, constraint in enumerate(A):
        constraint_string = '+'.join(str(constraint[j]) + f'x{j+1}' for j in range(len(constraint)))
        constraints.append(constraint_string + constraint_types[i] + str(b[i]))
    bnb = BranchAndBound(objective_function_string, constraints, Objective.MAX.value if tipo_problema == 'max' else Objective.MIN.value)
    # try:
    solution, best_solution, best_value = bnb.solve(False)
    print(best_solution, best_value)
    edges = [{'from': i, 
              'to': solution[i]['parent_id'] if solution[i]['parent_id'] != None else 'null',
              'label': solution[i]['simplex'].restriction_strings[-1],
              } for i in range(len(solution))]
    
    nodes = [{'id': i,
              'label': f'{solution[i]["solution"]["solucao"]["solucao"]}',
              'title': f'Z = {solution[i]["solution"]["solucao"]}<p>{solution[i]["simplex"].basic_vars}', 
              'level':solution[i]['level'], 
              'group': get_group(solution, best_value, i)[0],
              'content': templates.TemplateResponse(
                request=request, name="card_problem.html", context={"problema": solution[i], "class": get_group(solution, best_value, i)[1]}
              ).body
              } for i in range(len(solution))]
    node_otimo = [node['id'] for node in nodes if node['group'] == 'best'][0]
    return templates.TemplateResponse(
        request=request, name="branch_and_bound_solver.html", context={"solution": solution, "edges": edges, "nodes": nodes, "node_otimo": node_otimo}
    )
    
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
