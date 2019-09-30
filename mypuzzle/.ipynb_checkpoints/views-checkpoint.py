from flask import render_template
from flaskexample import app

@app.route('/')
@app.route('/index')
def index():
   user = { 'nickname': 'Miguel' } # fake user
   return render_template("index.html",
       title = 'Home',
       user = user)

@app.route('/test')
def test():
   user = { 'nickname': 'Miguel' } # fake user
   return render_template("test.html",
       title = 'Home',
       user = user)