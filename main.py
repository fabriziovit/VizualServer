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
app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])


class Converter:
    # get the image building the image from the tiles read at the maximum level possible
    def get_zoom(self, slide_filename, tile_size=4096):
        slide_file = OpenSlide(slide_filename)
        resize_value = 512
        resize_tile_div_value = tile_size / resize_value
        col_ini = 0
        row_ini = 0
        dz = DeepZoomGenerator(slide_file, tile_size, 1, False)
        col_fin = math.ceil(slide_file.dimensions[0] / tile_size)
        row_fin = math.ceil(slide_file.dimensions[1] / tile_size)

        new_im = Image.new('RGB', (int((slide_file.dimensions[0] / tile_size) * resize_value),
                                   int((slide_file.dimensions[1] / tile_size) * resize_value)))
        y_offset = 0
        x_offset = 0
        for i in range(col_ini, col_fin):
            for j in range(row_ini, row_fin):
                tile = dz.get_tile(dz.level_count - 1, (i, j))
                tile = tile.resize(
                    (int(tile.size[0] / resize_tile_div_value), int(tile.size[1] / resize_tile_div_value)),
                    Image.ANTIALIAS)
                new_im.paste(tile, (x_offset, y_offset))
                y_offset += resize_value
            y_offset = 0
            x_offset += resize_value
        return new_im

'''
    # get the image using the level dimension(not used)
    def get_zoom_2(self, slide_filename, tile_size=8192):
        slide_file = OpenSlide(slide_filename)
        dz = DeepZoomGenerator(slide_file, tile_size, 1, False)
        cont = dz.level_count
        print("Converting image...")
        for i in reversed(dz.level_dimensions):
            cont = cont - 1
            print(i[0], i[1])
            if i[0] <= 6000 and i[1] <= 6000:
                print(cont)
                break
        image = dz.get_tile(cont, (0, 0))  # image size: (8000, 8193) invece (8001, 9619)
        return image
'''

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
            image.save(img_io, 'JPEG')
            print("Converted.")
            return send_file(img_io, mimetype='image/jpeg')
    except:
        logger.error("Error compressing image")
        return Response(status=500)


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
    max_value = 8000
    tile_size = 4096
    resize_value = 512
    values = images.get(name)
    [name, _] = values
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path + ".svs")
    levels = openslide.dimensions
    image_width = levels[0]

    new_width = int((image_width / tile_size) * resize_value)
    ratio = new_width / image_width
    left = int(left / ratio)
    top = int(top / ratio)
    height = int(height / ratio)
    width = int(width / ratio)
    image = openslide.read_region((left, top), 0, (width, height))
    image = image.convert('RGB')
    img_io = image_abs_path + "_cropped.jpeg"
    ratio_crop = min(max_value / image.width, max_value / image.height)
    if ratio_crop < 1:
        image = image.resize((math.floor(image.width * ratio_crop), math.floor(image.height * ratio_crop)),
                             Image.ANTIALIAS)
    image.save(img_io, 'JPEG')
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
        image.save(img_io)
        return send_file(img_io, mimetype='image/jpeg')
    elif os.path.exists(image_abs_path + ".jpg"):
        image = Image.open(image_abs_path + ".jpg")
        image = image.convert('L')
        img_io = image_abs_path + "_grayscaled.jpeg"
        image.save(img_io)
        return send_file(img_io, mimetype='image/jpeg')
    converter = Converter()
    image = converter.get_zoom(image_abs_path + ".svs")
    image = image.convert('L')
    img_io = image_abs_path + "_grayscaled.jpeg"
    image.save(img_io)
    return send_file(img_io, mimetype='image/jpeg')


# get grayscale version of the cropped image already requested
@app.route('/api/get-image-cropped-grayscale/<name>/<int:left>_<int:top>_<int:width>x<int:height>')
def get_image_cropped_grayscale(name, left, top, width, height):
    max_value = 8000
    tile_size = 4096
    resize_value = 512
    values = images.get(name)
    [name, _] = values
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
