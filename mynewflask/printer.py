from flask import render_template

def print_it(name=None):
    return render_template('page2.html', name=name)

print_it()