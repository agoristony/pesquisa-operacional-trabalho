from flask import Blueprint, render_template, request

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
        return render_template("entrada.html", variaveis = variaveis, restricoes = restricoes, metodo=metodo)

