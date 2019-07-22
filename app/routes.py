from flask import jsonify, request, render_template, flash
from genreit import classify_movie
from app import app, model
from app.forms import DataForm

FEATURES = {'title': 'Required',
            'description': 'Required'}


def parse_args(req):
    args = {}
    for feature, default in FEATURES.items():
        value = req.get(feature, None)
        if value:
            args[feature] = value
        else:
            if default == 'Required':
                raise ValueError(f'"{feature}" must be provided.')
            else:
                args[feature] = default

    return args


@app.route('/api', methods=['GET'])
def api():
    if not request.json:
        return jsonify({'ERROR': 'No request received.'})

    args = parse_args(request.json)
    preds = classify_movie(args['title'], args['description'], model=model, verbose=False)

    return jsonify({'genres': preds})


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = DataForm()
    error = False

    if form.validate_on_submit():
        # Use these for input checking
        title = form.title.data
        description = form.description.data
        preds = classify_movie(title, description, model=model, verbose=False)
        error = True

        for pred in preds:
            flash(pred)

    return render_template('index.html', error=error, form=form)