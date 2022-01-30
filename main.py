from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import os
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])


@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    image_path = images.get(image_name)
    if image_name is not None and image_path is not None:
        try:
            real_path = os.path.dirname(os.path.realpath(__file__))
            image_abs_path = os.path.join(real_path + "\\images", image_path)
            if os.path.splitext(image_path)[1] == '.tif':
                img = Image.open(image_abs_path)
                img_io = BytesIO()
                img.save(img_io, 'JPEG', quality=1)
                img_io.seek(0)
                return send_file(img_io, mimetype='image/jpeg')
            else:
                return send_file(image_abs_path, mimetype='image/jpg')
        except:
            logger.error("Error compressing image")
            return Response(status=500)
    return Response(status=400)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
