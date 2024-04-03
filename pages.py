from flask import Blueprint, render_template, request
import models

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
        print(solver)
        return render_template("entrada.html", solver=solver)

@bp.route("/entrada_dados", methods=["GET","POST"])
def entrada_dados():
    if request.method == "POST":
        print(request.form)
    return "OK"