from flask import Flask, jsonify, Response, send_file
from flask_limiter import Limiter
from flask import request
from flask_limiter.util import get_remote_address
from middlewares.authorization import AuthorizationMiddleware
from ListImages import images
import os
os.add_dll_directory('C:/Users/Fabrizio/AppData/Local/Programs/Python/Python39/Lib/openslide-win64-20171122/bin')
from PIL import Image
import logging
from openslide import OpenSlide


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

@app.route('/api/get-image', methods=['POST'])
def get_image():
    image_name = request.form.get("name")
    image_path = images.get(image_name)
    if image_name is not None and image_path is not None:
            real_path = os.path.dirname(os.path.realpath(__file__))
            image_abs_path = os.path.join(real_path + "/images", image_path)
            print(real_path, image_abs_path)
            if os.path.splitext(image_path)[1] == '.svs':
                UNIT_X, UNIT_Y = 15000, 15000
                try:
                    save_name = image_name.split(".")[0] + ".jpg"
                    print("Processing : %s" % image_name)
                    os_obj = OpenSlide(image_abs_path)
                    w, h = os_obj.dimensions
                    w_rep, h_rep = int(w / UNIT_X) + 1, int(h / UNIT_Y) + 1
                    w_end, h_end = w % UNIT_X, h % UNIT_Y
                    w_size, h_size = UNIT_X, UNIT_Y
                    w_start, h_start = 0, 0
                    v_lst = []
                    print(w, h)
                    print("AAAAA")
                    for i in range(h_rep):
                        if i == h_rep - 1:
                            h_size = h_end
                        h_lst = []
                        for j in range(w_rep):
                            if j == w_rep - 1:
                                w_size = w_end
                            img = os_obj.read_region((w_start, h_start), 0, (w_size, h_size))
                            img = img.convert("RGB")
                            h_lst.append(img)
                            w_start += UNIT_X
                        v_lst.append(h_lst)
                        w_size = UNIT_X
                        h_start += UNIT_Y
                        w_start = 0
                    concat_h = [_get_concat_h(v) for v in v_lst]
                    print("BBBBB")
                    #stampa BBBB poi da errore prima c'era save_name da provare con image_name
                    concat_hv = _get_concat_v(concat_h)
                    concat_hv.save(image_path + "\\" + image_name)
                    print("CCCCC")
                    return send_file(image_path + "\\" + image_name, mimetype='image/jpg')
                except:
                    logger.error("Error compressing image")
                    return Response(status=500)
    return Response(status=400)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
