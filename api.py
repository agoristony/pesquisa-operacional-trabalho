from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from simplex import Simplex, Objective
from utils import Table

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
    tipoSimplex = form['tipoSimplex']
    tipoProblema = form['tipoProblema']
    return templates.TemplateResponse(
        request=request, name="form_simplex.html", context={"variaveis": variaveis, "tipoSimplex": tipoSimplex, "tipoProblema": tipoProblema}
    )
    
@app.post("/simplex_solver", response_class=HTMLResponse)
async def read_item(request: Request):
    form = await request.form()
    keys = form.keys()
    num_vars = len([key for key in keys if key.startswith('a1')])
    num_constraints = len([key for key in keys if key.startswith('b')])
    constraint_types = [form[f'relacao{i}'] for i in range(1, num_constraints + 1)]
    b = [float(form[f'b{i}']) for i in range(1, num_constraints + 1)]
    A = [[float(form[f'a{i}{j}']) for j in range(1, num_vars + 1)] for i in range(1, len(b) + 1)]
    objective_function = [form[f'c{i}'] for i in range(1, num_vars + 1)]
    objective_function_string = '+'.join(objective_function[i-1] + f'x{i}' for i in range(1, num_vars + 1))
    simplex = Simplex(objective_function_string, Objective.MIN.value)
    for i, constraint in enumerate(A):
        constraint_string = '+'.join(str(constraint[j]) + f'x{j+1}' for j in range(len(constraint)))
        print(constraint_string + constraint_types[i] + str(b[i]))
        simplex.add_restriction(constraint_string + constraint_types[i] + str(b[i]))
    simplex.table = Table.normalize_table(simplex.objective_function, simplex.table, simplex.column_b)
    solution = simplex.solve(False)
    return templates.TemplateResponse(
        request=request, name="simplex_solver.html", context={"solution": solution}
    )
    
        