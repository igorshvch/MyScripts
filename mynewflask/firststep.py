from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/print')
def print_it(name=None):
    return render_template('page.html', name=name)