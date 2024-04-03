from flask import Blueprint, render_template, request
import models
import sympy as sym

bp = Blueprint("pages", __name__)

@bp.route("/")
def home():
    return render_template("index.html")

@bp.route("/entrada", methods=["GET","POST"])
def entrada():
    if request.method == "POST":
        metodo = request.form["metodo"]
        variaveis = int(request.form["num_variaveis"])
        restricoes = int(request.form["num_restricoes"])
        problema = models.Problema("Problema", [models.Variavel(f"X{i}") for i in range(variaveis)], [models.Restricao(f"R{i}", [models.Variavel(f"x{j}") for j in range(variaveis)]) for i in range(restricoes)])
        solver = models.Solver(problema, metodo)
        return render_template("entrada.html", solver=solver)

@bp.route("/entrada_dados", methods=["GET","POST"])
def entrada_dados():
    if request.method == "POST":
        #ImmutableMultiDict([('modo', '1'), ('objetivoX0', '0'), ('objetivoX1', '0'), ('restricaoR0-X0', '0'), ('restricaoR0-X1', '0'), ('resultadoR0', '0'), ('restricaoR1-X0', '0'), ('restricaoR1-X1', '0'), ('resultadoR1', '0')])
        modo = int(request.form["modo"])
        metodo = request.form["metodo"]
        num_variaveis = int(request.form["num_variaveis"])
        num_restricoes = int(request.form["num_restricoes"])
        objetivo = [int(request.form[f"objetivoX{i}"]) for i in range(num_variaveis)]
        restricoes = []
        for i in range(num_restricoes):
            restricoes.append(models.Restricao(f"R{i}", [models.Variavel(f"X{j}") for j in range(num_variaveis)], coeficientes_variaveis=[int(request.form[f"restricaoR{i}-X{j}"]) for j in range(num_variaveis)], operador=int(request.form[f"operadorR{i}"]), valor=int(request.form[f"resultadoR{i}"])) )
        problema = models.Problema("Problema", [models.Variavel(f"X{i}") for i in range(num_variaveis)], restricoes, modo)
        problema.funcaoObjetivo = objetivo
    # return the objective function and the restrictions in the form of a problem
    return [str(restricao) for restricao in problema.restricoes]