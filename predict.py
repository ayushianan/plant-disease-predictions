from flask import Flask, request, render_template, send_from_directory
from fastai.vision import *
import numpy as np

path = Path('data/plantdatafull')
learn = load_learner(path)

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'data/plantdatafull/images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
        img = open_image(destination)
        pred_class,pred_idx,outputs = learn.predict(img)

    return render_template('report.html',pred_class=pred_class)


if __name__ == '__main__':
    app.run(debug=True)
