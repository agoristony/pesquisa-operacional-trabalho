from flask import Blueprint, render_template

bp = Blueprint("pages", __name__)

@bp.route("/")
def home():
    return render_template("index.html")

@bp.route("/entrada")
def entrada():
    return render_template("entrada.html", variaveis = 3, restricoes = 2)
