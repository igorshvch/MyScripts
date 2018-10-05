import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from eng1.db import get_db

bp = Blueprint('create_test', __name__, url_prefix='/create_test')

@bp.route('/create', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        testtext = request.form['testtext']
        db = get_db()
        error = None

        if not testtext:
            error = 'Test text is required.'
        if error is None:
            db.execute('INSERT INTO test (body) VALUES (?)',(testtext,))
            db.commit()
            return redirect(url_for('hello'))

        flash(error)

    return render_template('main.html')