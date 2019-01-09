# -*- coding:utf-8 -*-
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from mnist import Mnist
import json
from io import BytesIO

app = Flask(__name__)
CORS(app)

mn = Mnist()


@app.route("/", methods=['POST'])
def index():
    img = request.files.get("img_photo")
    byte_io = BytesIO(img.read())
    img = Image.open(byte_io)
    reco_result = mn.get_pic_number(img)
    del byte_io
    del img
    return json.dumps({"result": str(reco_result[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
