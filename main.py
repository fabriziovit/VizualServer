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

logger = logging.getLogger(__name__)
app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["20/minute"])
app.wsgi_app = AuthorizationMiddleware(app.wsgi_app)


@app.route('/api/get-images', methods=['GET'])
def get_images_names():
    return jsonify(data=[*images])


class Converter:
    @staticmethod
    def get_chunk(x_tile_num, tile_size, slide_file, iy, slide_width, slide_height, slide_img):
        for ix in range(x_tile_num):
            start_x = ix * tile_size
            len_x = tile_size if (ix + 1) * tile_size < slide_width else (slide_width - start_x)
            start_y = iy * tile_size
            len_y = tile_size if (iy + 1) * tile_size < slide_height else (slide_height - start_y)
            cur_tile = slide_file.read_region(location=(start_x, start_y), level=0, size=(len_x, len_y))
            slide_img[start_y:start_y + len_y, start_x:start_x + len_x, :] = np.array(cur_tile)[:, :, :3]

    def save_svs_img(self, slide_filename, tile_size=8192):
        slide_file = OpenSlide(slide_filename)
        slide_width, slide_height = slide_file.dimensions
        slide_img = np.zeros((slide_height, slide_width, 3), np.uint8)
        x_tile_num = int(np.floor((slide_width - 1) / tile_size)) + 1
        y_tile_num = int(np.floor((slide_height - 1) / tile_size)) + 1
        threads = list()
        for iy in range(y_tile_num):
            thread = Thread(target=self.get_chunk,
                            args=(x_tile_num, tile_size, slide_file, iy, slide_width, slide_height, slide_img))
            threads.append(thread)
            thread.start()

        for index, thread in enumerate(threads):
            thread.join()

        print("Converting...")

        return Image.fromarray(slide_img)



@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    image_path = images.get(image_name)
    if image_name is not None and image_path is not None:
        try:
            real_path = os.path.dirname(os.path.realpath(__file__))
            image_abs_path = os.path.join(real_path + "/images", image_path)
            print(real_path, image_abs_path)
            converter = Converter()
            image = converter.save_svs_img(image_abs_path)
            img_io = BytesIO()
            image.save(img_io, 'JPEG', quality=65)
            #img_io.seek(0)
            print("Converted.")
            return send_file(img_io, mimetype='image/jpeg')
        except:
            logger.error("Error compressing image")
            return Response(status=500)
        # if os.path.splitext(image_path)[1] == '.svs':
    return Response(status=400)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
    '''
    real_path = os.path.dirname(os.path.realpath(__file__))
    image_abs_path = os.path.join(real_path + "/images/test1.svs")
    image_out_path = os.path.join(real_path + "/images/test.jpg")
    converter = Converter()
    converter.save_svs_img(image_abs_path, image_out_path)
    '''
