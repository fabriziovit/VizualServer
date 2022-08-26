import math

from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import numpy as np
import os

# os.add_dll_directory('C:/Users/Fabrizio/AppData/Local/Programs/Python/Python39/Lib/openslide-win64-20171122/bin')
import signal
from PIL import Image
import logging
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

logger = logging.getLogger(__name__)
app = Flask(__name__)
# limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
#app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])


class Converter:
    # get the image building the image from the tiles read at the maximum level possible
    def get_zoom(self, slide_filename):
        max_width = 5120
        max_height = 3007
        slide_file = OpenSlide(slide_filename)
        coeff = max(slide_file.dimensions[1] / max_height, slide_file.dimensions[0] / max_width)
        width_result = slide_file.dimensions[0] / coeff
        height_result = slide_file.dimensions[1] / coeff
        image = slide_file.get_thumbnail((width_result, height_result))
        print(image.size)
        return image


# get the image requested send the image if is already been converted and stored, otherwise create the image from the .svs file
@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    values = images.get(image_name)
    [image_path, _] = values
    try:
        real_path = os.path.dirname(os.path.realpath(__file__))
        image_abs_path = os.path.join(real_path + "/images", image_path)
        print(image_abs_path)
        if os.path.exists(image_abs_path + ".jpeg"):
            return send_file(image_abs_path + ".jpeg", mimetype='image/jpeg')
        elif os.path.exists(image_abs_path + ".jpg"):
            return send_file(image_abs_path + ".jpg", mimetype='image/jpg')
        else:
            print("Converting image...")
            converter = Converter()
            image = converter.get_zoom(image_abs_path + ".svs")
            img_io = image_abs_path + ".jpeg"
            image.save(img_io, 'JPEG', quality=100)
            print("Converted.")
            return send_file(img_io, mimetype='image/jpeg')
    except:
        logger.error("Error compressing image")
        return Response(status=500)

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


# returns the ratio of the image(not used)
def get_ratio(file_name, level):
    selected_size = 0
    slide_file = OpenSlide(file_name + ".svs")
    dz = DeepZoomGenerator(slide_file, 8192, pow(2, 14), False)
    original_size = dz.level_dimensions[len(dz.level_dimensions) - 1]
    cont = len(dz.level_dimensions) - 1
    level = len(dz.level_dimensions) - (level - 1)
    print(level, "    ", cont)
    tiles = dz.tile_count
    print(tiles)
    for i in reversed(dz.level_dimensions):
        if cont == level:
            selected_size = i
            break
        cont -= 1
    ratio = min(selected_size[0] / original_size[0], selected_size[1] / original_size[1])
    return ratio


# returns the level used from the image for the preview(not used)
def get_level(file_name):
    slide_file = OpenSlide(file_name + ".svs")
    dz = DeepZoomGenerator(slide_file, 8192, 1, False)
    cont = 0
    for i in reversed(dz.level_dimensions):
        cont += 1
        if i[0] <= 6000 and i[1] <= 6000:
            break
    return cont


# get the 2 coords and the width and the height of the image cropped also the name of the image, the function calculate the ratio of the area cropped and return the image
@app.route('/api/get-image-cropped/<name>/<int:left>_<int:top>_<int:width>x<int:height>')
def get_image_cropped(name, left, top, width, height):
    max_width = 5120
    max_height = 3007
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    levels = openslide.dimensions
    image_width = levels[0]
    image_height = levels[1]
    if width < 0 or width > image_width:
        return Response(status=500)
    if height < 0 or height > image_height:
        return Response(status=500)
    image = openslide.read_region((left, top), 0, (width, height))
    image = image.convert('RGB')
    img_io = image_abs_path + "_cropped"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpeg"
    ratio_crop = max(image.width / max_width, image.height / max_height)
    if ratio_crop > 1:
        image = image.resize((math.floor(image.width / ratio_crop), math.floor(image.height / ratio_crop)),
                             Image.ANTIALIAS)
    image.save(img_io, 'JPEG', quality=100)
    return send_file(img_io, mimetype='image/jpeg')


# get the image with the grayscale filter
@app.route('/api/get-image-grayscale/<name>')
def get_image_grayscale(name):
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    if os.path.exists(image_abs_path + "_grayscaled.jpeg"):
        return send_file(image_abs_path + "_grayscaled.jpeg", mimetype='image/jpeg')
    if os.path.exists(image_abs_path + ".jpeg"):
        image = Image.open(image_abs_path + ".jpeg")
        image = image.convert('L')
        img_io = image_abs_path + "_grayscaled.jpeg"
        image.save(img_io, quality=100)
        return send_file(img_io, mimetype='image/jpeg')
    elif os.path.exists(image_abs_path + ".jpg"):
        image = Image.open(image_abs_path + ".jpg")
        image = image.convert('L')
        img_io = image_abs_path + "_grayscaled.jpeg"
        image.save(img_io, quality=100)
        return send_file(img_io, mimetype='image/jpeg')
    converter = Converter()
    image = converter.get_zoom(image_abs_path + ".svs")
    image = image.convert('L')
    img_io = image_abs_path + "_grayscaled.jpeg"
    image.save(img_io, quality=100)
    return send_file(img_io, mimetype='image/jpeg')


# get grayscale version of the cropped image already requested
@app.route('/api/get-image-cropped-grayscale/<name>/<int:left>_<int:top>_<int:width>x<int:height>')
def get_image_cropped_grayscale(name, left, top, width, height):
    max_width = 5120
    max_height = 3007
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    if os.path.exists(image_abs_path + "_cropped"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpeg"):
        image = Image.open(image_abs_path + "_cropped"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpeg")
        image = image.convert('L')
        img_io = image_abs_path + "_grayscaled"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpeg"
        image.save(img_io, quality=100)
    if os.path.exists(image_abs_path + "_cropped"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpg"):
        image = Image.open(image_abs_path + "_cropped"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpg")
        image = image.convert('L')
        img_io = image_abs_path + "_grayscaled"+str(left)+"x"+str(top)+"x"+str(width)+"x"+str(height)+".jpeg"
        image.save(img_io, quality=100)
        return send_file(img_io, mimetype='image/jpeg')
    openslide = OpenSlide(image_abs_path + ".svs")
    levels = openslide.dimensions
    image_width = levels[0]
    image_height = levels[1]
    if left < 0 or left > image_width:
        return Response(status=500)
    if top < 0 or top > image_height:
        return Response(status=500)
    if width < 0 or width > image_width:
        return Response(status=500)
    if height < 0 or height > image_height:
        return Response(status=500)
    image = openslide.read_region((left, top), 0, (width, height))
    image = image.convert('RGB').convert('L')
    img_io = image_abs_path + "_cropped_grayscaled.jpeg"
    ratio_crop = max(image.width / max_width, image.height / max_height)
    if ratio_crop < 1:
        image = image.resize((math.floor(image.width * ratio_crop), math.floor(image.height * ratio_crop)),
                             Image.ANTIALIAS)
    image.save(img_io, quality=100)
    return send_file(img_io, mimetype='image/jpeg')


# remove files that ends with .jpeg and .jpg from the director that contains the images
def exit_handler(code, frame):
    real_path = os.path.dirname(os.path.realpath(__file__))
    for fil in os.listdir(real_path + "/images"):
        if fil.endswith('.jpeg') or fil.endswith('.jpg'):
            os.remove(real_path + "/images/" + fil)
    print("SPEGNIMENTO SERVER... CANCELLAZIONE FILES CONVERTITI.")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_handler)
    app.run(host='127.0.0.1', port=8000, debug=True)
