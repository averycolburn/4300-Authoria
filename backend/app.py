import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from authoria import Authoria

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# going off the cocktail lab's app.py implementation
# # These are the DB credentials for your OWN MySQL
# # Don't worry about the deployment credentials, those are fixed
# # You can use a different DB name if you want to
# LOCAL_MYSQL_USER = "root"
# LOCAL_MYSQL_USER_PASSWORD = "admin"
# LOCAL_MYSQL_PORT = 3306
# LOCAL_MYSQL_DATABASE = "kardashiandb"

# mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

authoria = Authoria()

#again going off of cocktail lab's app.py implementation
# # Sample search, the LIKE operator in this case is hard-coded, 
# # but if you decide to use SQLAlchemy ORM framework, 
# # there's a much better and cleaner way to do this
# def sql_search(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id","title","descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])

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
    # return sql_search(text)

    # max_pop = -1
    # max_i = 0
    res = authoria.query_svd(text)
    # for i in range(len(res)):
    #     if res[i]['rating'] > max_pop:
    #         max_pop = res[i]['rating']
    #         max_i = i

    # for i in range(len(res)):
    #     if i == max_i:
    #         res[i] = dict(res[i])
    #     else:
    #         res[i] = dict(res[i])
    j = json.dumps(res, default = set_default)
    return j

@app.route("/query_pop")
def episodes_search_pop():
    text = request.args.get("description")
    # return sql_search(text)

    # max_pop = -1
    # max_i = 0
    res = authoria.query_svd(text, True)
    # for i in range(len(res)):
    #     if res[i]['rating'] > max_pop:
    #         max_pop = res[i]['rating']
    #         max_i = i

    # for i in range(len(res)):
    #     if i == max_i:
    #         res[i] = dict(res[i])
    #     else:
    #         res[i] = dict(res[i])
    j = json.dumps(res, default = set_default)
    return j

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)