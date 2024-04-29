import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from authoria import Authoria

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

app = Flask(__name__)
CORS(app)

authoria = Authoria()


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# from https://stackoverflow.com/questions/22281059/set-object-is-not-json-serializable
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


@app.route("/query_rel")
def episodes_search():
    text = request.args.get("description")

    res = authoria.query_svd(text)
    j = json.dumps(res, default = set_default)
    return j

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)