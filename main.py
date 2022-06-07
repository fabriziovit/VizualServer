from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import numpy as np
import os
os.add_dll_directory('C:/Users/Fabrizio/AppData/Local/Programs/Python/Python39/Lib/openslide-win64-20171122/bin')
import signal
from PIL import Image
import logging
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

logger = logging.getLogger(__name__)
app = Flask(__name__)
#limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)

@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])

class Converter:
    def get_zoom(self, slide_filename, tile_size=8192):
        slide_file = OpenSlide(slide_filename)
        dz = DeepZoomGenerator(slide_file, tile_size, 1, False)
        cont = dz.level_count
        for i in reversed(dz.level_dimensions):
            cont = cont - 1
            print(i[0], i[1])
            if i[0] <= 6000 and i[1] <= 6000:
                print(cont)
                break
        print("qui")
        image = dz.get_tile(cont, (0, 0))# image size: (8000, 8193) invece (8001, 9619)
        print("quo")
        return image

#trovare altre immagini svs per provare
@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    image_path = images.get(image_name)
    try:
        real_path = os.path.dirname(os.path.realpath(__file__))
        image_abs_path = os.path.join(real_path + "/images", image_path)
        print(image_abs_path)
        if os.path.exists(image_abs_path + ".jpeg"):
            return send_file(image_abs_path+ ".jpeg", mimetype='image/jpeg')
        elif os.path.exists(image_abs_path + ".jpg"):
            return send_file(image_abs_path + ".jpg", mimetype='image/jpg')
        else:
            converter = Converter()
            image = converter.get_zoom(image_abs_path+".svs")
            img_io = image_abs_path+".jpeg"
            image.save(img_io, 'JPEG')
            print("Converted.")
            return send_file(img_io, mimetype='image/jpg')
    except:
        logger.error("Error compressing image")
        return Response(status=500)

#funzione che restituisce il ratio dell'immagine
#da usare se il ratio di un immagine nel json è uguale a 0
def get_ratio(file_name):
    original_size = 0
    selected_size = 0
    slide_file = OpenSlide(file_name+".svs")
    dz = DeepZoomGenerator(slide_file, 8192, 1, False)
    cont = dz.level_count
    for i in reversed(dz.level_dimensions):
        original_size = i
        cont = cont - 1
        if i[0] <= 6000 and i[1] <= 6000:
            selected_size = i
            break
    ratio = min(selected_size[0]/original_size[0], selected_size[1]/original_size[1])
    return ratio

#get the 4 coords of the image and the name of the image and the ratio between the resized and the original image
@app.route('/api/get-image-cropped/<name>/<int:left>_<int:top>_<int:width>_<int:height>/<int:ratio>')
def get_image_cropped(name, left, top, width, height, ratio):
    #name = da json get name
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images", name)
    openslide = OpenSlide(image_abs_path+".svs")
    left = left/ratio
    top = top/ratio
    width = width/ratio
    height = height/ratio
    image = openslide.read_region((left, top), 0, (width, height))#da provare
    img_io = image + "-cropped.jpeg"
    image.save(img_io, 'JPEG')
    return send_file(img_io, mimetype='image/jpg')

def exit_handler(code, frame):
    # cancellazione files che finiscono con .jpeg
    real_path = os.path.dirname(os.path.realpath(__file__))
    for fil in os.listdir(real_path+"/images"):
        if fil.endswith('.jpeg') or fil.endswith('.jpg'):
            os.remove(real_path+"/images/"+fil)
    print("SPEGNIMENTO SERVER... CANCELLAZIONE FILES CONVERTITI.")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_handler)
    app.run(host='127.0.0.1', port=8000, debug=True)