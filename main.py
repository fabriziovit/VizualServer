from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import numpy as np
from io import BytesIO
import os
os.add_dll_directory('C:/Users/Fabrizio/AppData/Local/Programs/Python/Python39/Lib/openslide-win64-20171122/bin')
from PIL import Image
import logging
from threading import Thread
from openslide import OpenSlide
import time


logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])

def _get_concat_h(img_lst):
    width, height, h = sum([img.width for img in img_lst]), img_lst[0].height, 0
    dst = Image.new('RGB', (width, height))
    for img in img_lst:
        dst.paste(img, (h, 0))
        h += img.width
    return dst

def _get_concat_v(img_lst):
    width, height, v = img_lst[0].width, sum([img.height for img in img_lst]), 0
    dst = Image.new('RGB', (width, height))
    for img in img_lst:
        dst.paste(img, (0, v))
        v += img.height
    return dst

class IlMioThread (Thread):
   def __init__(self, image_path, durata):
       Thread.__init__(self)
       self.durata = durata
       save_svs_img(image_path)
   def run(self):
       print("Thread avviato")
       time.sleep(self.durata)
       print("Thread terminato")

def save_svs_img(slide_filename, tile_size=8192):
    slide_file = OpenSlide(slide_filename)
    slide_width, slide_height = slide_file.dimensions
    # tile_arr = []
    slide_img = np.zeros((slide_height, slide_width, 3), np.uint8)
    x_tile_num = int(np.floor((slide_width-1)/tile_size)) + 1
    y_tile_num = int(np.floor((slide_height-1)/tile_size)) + 1
    for iy in range(y_tile_num):
        for ix in range(x_tile_num):
            start_x = ix * tile_size
            len_x = tile_size if (ix + 1) * tile_size < slide_width else (slide_width - start_x)
            start_y = iy * tile_size
            len_y = tile_size if (iy + 1) * tile_size < slide_height else (slide_height - start_y)
            # tile_arr.append(((start_x, start_y), (len_x, len_y)))
            cur_tile = slide_file.read_region(location=(start_x, start_y), level=0, size=(len_x, len_y))
            slide_img[start_y:start_y+len_y, start_x:start_x+len_x, :] = np.array(cur_tile)[:,:,:3]
    slide_savename = os.path.splitext(slide_filename)[0] + '.tif'
    #da vedere se funnziona così
    img = Image.open(slide_savename)
    img_io = BytesIO()
    print("1111111")
    img.save(img_io, 'JPEG', quality=1)
    img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    print("AAAAAAAAAA")
    return send_file(img_io, mimetype='image/jpeg')
    #io.imsave(slide_savename, slide_img


@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    image_path = images.get(image_name)
    if image_name is not None and image_path is not None:
            real_path = os.path.dirname(os.path.realpath(__file__))
            image_abs_path = os.path.join(real_path + "/images", image_path)
            print(real_path, image_abs_path)
            #if os.path.splitext(image_path)[1] == '.svs':
            thread1 = IlMioThread(image_abs_path, 240)

            thread1.start()
            thread1.join()
    return Response(status=400)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
