import math

from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import os

os.add_dll_directory('C:/Users/Fabrizio/AppData/Local/Programs/Python/Python39/Lib/openslide-win64-20171122/bin')
import signal
from PIL import Image
import logging
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

logger = logging.getLogger(__name__)
app = Flask(__name__)


# limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
# app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])


class Converter:
    # get the image using the level dimension
    def get_zoom(self, slide_filename, tile_size=8192):
        max_width = 5120
        max_height = 3008
        slide_file = OpenSlide(slide_filename)
        coeff = max(slide_file.dimensions[1] / max_height, slide_file.dimensions[0] / max_width)
        width_result = slide_file.dimensions[0] / coeff
        height_result = slide_file.dimensions[1] / coeff
        print("Converting image...")
        image = slide_file.get_thumbnail((width_result, height_result))
        print(image.size)
        return image


@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    values = images.get(image_name)
    [image_path, _] = values
    try:
        real_path = os.path.dirname(os.path.realpath(__file__))
        image_abs_path = os.path.join(real_path + "/images", image_path)
        if os.path.exists(image_abs_path + ".jpeg"):
            return send_file(image_abs_path + ".jpeg", mimetype='image/jpeg')
        elif os.path.exists(image_abs_path + ".jpg"):
            return send_file(image_abs_path + ".jpg", mimetype='image/jpg')
        else:
            converter = Converter()
            image = converter.get_zoom(image_abs_path + ".svs")
            img_io = image_abs_path + ".jpeg"
            image.save(img_io, 'JPEG', quality=100)
            print("Converted.")
            return send_file(img_io, mimetype='image/jpeg')
    except:
        logger.error("Error compressing image")
        return Response(status=500)


def get_ratio(file_name):
    original_size = 0
    selected_size = 0
    slide_file = OpenSlide(file_name + ".svs")
    dz = DeepZoomGenerator(slide_file)
    cont = dz.level_count
    for i in reversed(dz.level_dimensions):
        original_size = i
        cont = cont - 1
        if i[0] <= 6000 and i[1] <= 6000:
            selected_size = i
            break
    ratio = min(selected_size[0] / original_size[0], selected_size[1] / original_size[1])
    return ratio


@app.route('/api/get-image-cropped/<name>/<int:left>_<int:top>_<int:width>x<int:height>')
def get_image_cropped_test(name, width, height, left, top):
    max_width = 5120
    max_height = 3008
    tile_size = 4096
    resize_value = 512
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    levels = openslide.dimensions
    image_width = levels[0]
    image_height = levels[1]
    '''
    ratio = max(image_height/max_height, image_width/max_width)
    print(image_height, image_width, ratio)
    left = int(left * ratio)
    if not (0 <= left <= image_width):
        left = image_width
    top = int(top * ratio)
    if not (0 <= top <= image_height):
        top = image_width
    width = int(width * ratio)
    if not ((left + width) < image_width):
        logger.error("Error cropping images, data not valid")
        return Response(status=500)
    height = int(height * ratio)
    if not ((top + height) < image_height):
        logger.error("Error cropping images, data not valid")
        return Response(status=500)
    '''
    image = openslide.read_region((left, top), 0, (width, height))
    image = image.convert('RGB')
    img_io = image_abs_path + "_cropped.jpeg"
    ratio_crop = max(image.width / max_width, image.height / max_height)
    if ratio_crop > 1:
        image = image.resize((math.floor(image.width / ratio_crop), math.floor(image.height / ratio_crop)),
                             Image.ANTIALIAS)
    image.save(img_io, 'JPEG', quality=100)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/api/get-image-width/<name>')
def get_image_width(name):
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    return jsonify(data=[openslide.dimensions[0]])


@app.route('/api/get-image-height/<name>')
def get_image_height(name):
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    return jsonify(data=[openslide.dimensions[1]])


@app.route('/api/get-image-grayscale/<name>')
def get_image_grayscale(name):
    values = images.get(name)
    [name, ratio] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    if os.path.exists(image_abs_path + "_grayscaled.jpeg"):
        return send_file(image_abs_path + "_grayscaled.jpeg", mimetype='image/jpeg')
    converter = Converter()
    image = converter.get_zoom(image_abs_path + ".svs")
    image = image.convert('L')
    img_io = image_abs_path + "_grayscaled.jpeg"
    image.save(img_io)
    return send_file(img_io, mimetype='image/jpeg')


# da modificare se funziona il metodo sopra
@app.route('/api/get-image-cropped-grayscale/<name>/<int:left>_<int:top>_<int:width>x<int:height>')
def get_image_cropped_grayscale(name, left, top, width, height):
    max_value = 8000
    tile_size = 4096
    resize_value = 512
    values = images.get(name)
    [name, ratio] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    levels = openslide.level_dimensions[0]
    imageW = levels[0]
    imageH = levels[1]
    new_width = int((imageW / tile_size) * resize_value)
    ratio = new_width / imageW
    left = int(left / ratio)
    if not (0 <= left <= imageW):
        left = imageW
    top = int(top / ratio)
    if not (0 <= top <= imageH):
        top = imageW
    width = int(width / ratio)
    if not ((left + width) < imageW):
        logger.error("Error cropping images, data not valid")
        return Response(status=500)
    height = int(height / ratio)
    if not ((top + height) < imageH):
        logger.error("Error cropping images, data not valid")
        return Response(status=500)
    image = openslide.read_region((left, top), 0, (width, height))
    image = image.convert('RGB').convert('L')
    img_io = image_abs_path + "_cropped_grayscaled.jpeg"
    ratio_crop = min(max_value / image.width, max_value / image.height)
    if ratio_crop < 1:
        image = image.resize((math.floor(image.width * ratio_crop), math.floor(image.height * ratio_crop)),
                             Image.ANTIALIAS)
    image.save(img_io)
    return send_file(img_io, mimetype='image/jpeg')


def exit_handler(code, frame):
    # cancellazione files che finiscono con .jpeg o .jpg
    real_path = os.path.dirname(os.path.realpath(__file__))
    for fil in os.listdir(real_path + "/images"):
        if fil.endswith('.jpeg') or fil.endswith('.jpg'):
            if fil == "test1.jpeg":
                break
            os.remove(real_path + "/images/" + fil)
    print("SPEGNIMENTO SERVER... CANCELLAZIONE FILES CONVERTITI.")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_handler)
    app.run(host='127.0.0.1', port=8000, debug=True)
