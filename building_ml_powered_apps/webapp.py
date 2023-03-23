from functools import lru_cache

from building_ml_powered_apps.second_model import get_pos_score_from_text
from building_ml_powered_apps.third_model import (
    get_recommendation_and_prediction_from_text,
    get_v3_model,
)
from flask import Flask, Request, abort, render_template, request

app = Flask(__name__)
v3_model = get_v3_model()


def model_from_template(template_name: str) -> str:
    return template_name.split(".")[0]


@lru_cache(maxsize=128)
def retreive_recommendations_for_model(question: str, model: str):
    if model == "v1":
        abort(400)
    if model == "v2":
        return get_pos_score_from_text(question)
    if model == "v3":
        return get_recommendation_and_prediction_from_text(v3_model, question)
    raise ValueError("Incorrect model received...")


def handle_text_request(request: Request, template_name: str):
    if request.method == "POST":
        question = request.form.get("question")
        model_name = model_from_template(template_name)
        suggestions = retreive_recommendations_for_model(
            question=question, model=model_name
        )
        payload = {
            "input": question,
            "suggestions": suggestions,
        }
        return render_template("results.html", ml_result=payload)
    else:
        return render_template(template_name)


@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/v3", methods=["POST", "GET"])
def v3():
    return handle_text_request(request, "v3.html")


@app.route("/v2", methods=["POST", "GET"])
def v2():
    return handle_text_request(request, "v2.html")


@app.route("/v1", methods=["POST", "GET"])
def v1():
    return abort(404, "Not Implemented")
