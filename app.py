from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap5

from flask_wtf import FlaskForm, CSRFProtect
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length

import tensorflow_hub as hub 
import tensorflow as tf
import numpy as np 

from backend.main import get_recommendations

import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

bootstrap = Bootstrap5(app)

csrf = CSRFProtect(app)

class FavouriteMovieForm(FlaskForm):
    text = TextAreaField("", validators=[DataRequired(), Length(5)])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = FavouriteMovieForm()
    message = ""
    if form.validate_on_submit():
        text = form.text.data

        return redirect(url_for("movie", text=text)) #todo, get movie by its id

    return render_template("index.html", form=form, message=message)

@app.route('/movie/<text>')
def movie(text):
    top_ten_rated_movies = get_recommendations(text)

    print(top_ten_rated_movies)

    return render_template("movie.html", top_ten_rated_movies=top_ten_rated_movies)


