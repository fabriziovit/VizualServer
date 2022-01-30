import numpy as np
import pickle as pkl
from multiprocessing import Pool
from keras.engine.training import Model
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.spatial.distance import cdist
from itertools import permutations
from os import listdir
from math import log
from bitstruct import *
#from skimage.metrics import structural_similarity as ssim



class PIFS:
    def __init__(self, w, h, rsize, code, features):
        self.w = w
        self.h = h
        self.rsize = rsize
        self.code = code
        self.features = features


class PIFS_WSI:
    def __init__(self, w, h, channels, range_size, tile_size, data):
        self.w = w
        self.h = h
        self.channels = channels
        self.range_size = range_size
        self.tile_size = tile_size
        self.data = data


class OPTS:
    def __init__(self, w, h, tilesize, range_size, domain_shift, num_dec_iter):
        self.w = w
        self.h = h
        self.tilesize = tilesize
        self.range_size = range_size
        self.domain_shift = domain_shift
        self.num_dec_iter = num_dec_iter


def my_imresize(d):
    n = d.shape[0]
    df = d.astype('float32')
    da = np.zeros((int(n + 2), int(n + 2)), dtype='float32')
    da[1: n + 1, 1: n + 1] = df
    da[0: n, 0: n] = da[0: n, 0: n] + df
    da[0: n, 1: n + 1] = da[0: n, 1: n + 1] + df
    da[1: n + 1, 0: n] = da[1: n + 1, 0: n] + df
    ds = np.divide(da[1: n + 1: 2, 1: n + 1: 2], 4.0)

    return ds


def hope_setup():
    return OPTS(2048, 1536, 512, 16, 8, 100)


# noinspection PyTypeChecker
def pifs_permute_domain_patches(c, pr):
    n = c.shape[0]
    n2 = round(n / 2)
    dp = np.zeros((int(n), int(n)), dtype='float32')
    t = [[1, 1, n2, n2], [1, n2 + 1, n2, n], [n2 + 1, 1, n, n2], [n2 + 1, n2 + 1, n, n]]

    dp[t[0][0] - 1: t[0][2], t[0][1] - 1: t[0][3]] = c[t[pr[0]][0] - 1: t[pr[0]][2], t[pr[0]][1] - 1: t[pr[0]][3]]
    dp[t[1][0] - 1: t[1][2], t[1][1] - 1: t[1][3]] = c[t[pr[1]][0] - 1: t[pr[1]][2], t[pr[1]][1] - 1: t[pr[1]][3]]
    dp[t[2][0] - 1: t[2][2], t[2][1] - 1: t[2][3]] = c[t[pr[2]][0] - 1: t[pr[2]][2], t[pr[2]][1] - 1: t[pr[2]][3]]
    dp[t[3][0] - 1: t[3][2], t[3][1] - 1: t[3][3]] = c[t[pr[3]][0] - 1: t[pr[3]][2], t[pr[3]][1] - 1: t[pr[3]][3]]

    n2 = n2 - 1
    dp[:, n2 - 1] = (dp[:, n2 - 2] + dp[:, n2 - 1] + dp[:, n2]) / 3.0
    dp[:, n2 + 1] = (dp[:, n2 - 1] + dp[:, n2] + dp[:, n2 + 1]) / 3.0
    dp[n2 - 1, :] = (dp[n2 - 2, :] + dp[n2 - 1, :] + dp[n2, :]) / 3.0
    dp[n2 + 1, :] = (dp[n2 - 1, :] + dp[n2, :] + dp[n2 + 1, :]) / 3.0

    return dp


def pifs_extract_domains_features(img, dsize, step, pr):
    w = img.shape[1]
    h = img.shape[0]
    nf = dsize * dsize / 4
    n_pr = len(pr)
    n = n_pr * len(range(0, h - dsize + 1, step)) * len(range(0, w - dsize + 1, step))
    features = np.zeros((int(n), int(nf)), dtype='float32')
    dlist = np.zeros((int(n), 4), dtype='float32')
    cnt = 0
    for i in range(0, h - dsize + 1, step):
        for j in range(0, w - dsize + 1, step):
            c = img[i:(i + dsize), j: (j + dsize)]
            for k in range(0, n_pr):
                d = pifs_permute_domain_patches(c, pr[int(k)])
                d = my_imresize(d)
                d = d.ravel()
                fv = (d - d.mean()) / d.std()
                dlist[cnt, :] = [i, j, dsize, k]
                features[cnt, :] = fv
                cnt = cnt + 1
    return dlist, features


def pifs_encode_wsi(a, tilesize, range_size, domain_shift):
    p = list(permutations(range(0, 4)))
    p.reverse()
    h = a.shape[0]
    w = a.shape[1]
    channels = a.shape[2]
    data = np.empty((int(h / tilesize), int(w / tilesize), channels), dtype=object)
    pg_sz = 64
    aprogress = np.zeros((pg_sz * len(range(0, h, tilesize)), pg_sz * len(range(0, w, tilesize)), channels),
                         dtype='float32')
    cnt_i = 0
    for i in range(0, h, tilesize):
        cnt_j = 0
        for j in range(0, w, tilesize):
            for k in range(0, channels):
                at = a[i:i + tilesize, j:j + tilesize, k]
                pifscode = pifs_encode_tile(at, range_size, domain_shift, p)
                data[cnt_i, cnt_j, k] = pifscode
                aprogress[pg_sz * cnt_i: pg_sz * (cnt_i + 1), pg_sz * cnt_j: pg_sz * (cnt_j + 1), k] = resize(at,
                                                                                                              [pg_sz,
                                                                                                               pg_sz],
                                                                                                              order=3,
                                                                                                              preserve_range=True,
                                                                                                              clip=False,
                                                                                                              anti_aliasing=True)
            cnt_j += 1
        cnt_i += 1
    return PIFS_WSI(w, h, channels, range_size, tilesize, data)


def pifs_decode_wsi(pcw: PIFS_WSI, unet: Model, zoom):
    p = list(permutations(range(0, 4)))
    p.reverse()
    niter = 100
    w = int(pcw.w * zoom)
    h = int(pcw.h * zoom)
    c = pcw.channels
    tilesize = int(pcw.tile_size * zoom)

    data = pcw.data

    thp = [i for i in range(0, h, tilesize)]
    twp = [i for i in range(0, w, tilesize)]

    b = np.zeros((h, w, c), dtype='float32')
    flag = np.zeros((tilesize, tilesize, c), dtype='float32')

    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            for k in range(0, c):
                flag[0: tilesize, 0:tilesize, k] = pifs_decode_tile(data[i, j, k], niter, zoom, p)

            if unet is not None:
                for x in range(0, tilesize, 256):
                    for y in range(0, tilesize, 256):
                        flagone = np.expand_dims(flag[x: x + 256, y: y + 256, :], axis=0)
                        flag[x: x + 256, y: y + 256, :] = unet.predict(flagone)
                # flag = np.argmax(flag[0], axis=-1)

            b[thp[i]: thp[i] + tilesize, twp[j]:twp[j] + tilesize, :] = flag[0:tilesize, 0:tilesize, :]

    b = NormalizeData(b) * 65535.0;
    return b.astype('uint16')


def pifs_encode_tile_old(img, rsize, step):
    pr = list(permutations(range(0, 4)))
    pr.reverse()
    h = img.shape[0]
    w = img.shape[1]
    dd = np.sqrt(w * w + h * h)
    dsize = 2 * rsize
    dlist, features = pifs_extract_domains_features(img, dsize, step, pr)

    nrgs = len(range(0, h - rsize + 1, rsize)) * len(range(0, w - rsize + 1, rsize))
    fv = np.zeros((int(nrgs), int(rsize * rsize)), dtype='float32')
    idxnn = np.zeros((int(nrgs)), dtype='float32')

    cnt = 0
    for i in range(0, h - rsize + 1, rsize):
        for j in range(0, w - rsize + 1, rsize):
            r = img[i:(i + rsize), j:(j + rsize)]
            r = r.ravel()
            fv[cnt, :] = np.array((r - r.mean()) / r.std())
            cnt += 1
    for i in range(0, nrgs, 256):
        dist = cdist(fv[i: i + 256], features, 'cityblock')
        idxnn[i: i + 256] = np.argmin(dist, axis=1)

    code = np.zeros((int(nrgs), 6), dtype='float32')
    features = np.zeros((int(nrgs), 4), dtype='float32')
    cnt = 0
    for i in range(0, h - rsize + 1, rsize):
        for j in range(0, w - rsize + 1, rsize):
            idd = int(idxnn[cnt])
            iss = int(dlist[idd, 3])
            yd = int(dlist[idd, 0])
            xd = int(dlist[idd, 1])

            r = img[i:(i + rsize), j:(j + rsize)]
            d = img[yd:(yd + dsize), xd:(xd + dsize)]
            d = pifs_permute_domain_patches(d, pr[iss])
            d = my_imresize(d)
            po = np.polyfit(d.ravel(), r.ravel(), 1)
            pzero = round(po[0] * 32) / 32
            puno = round(po[1] / 8) * 8
            code[cnt, 0:2] = [pzero, puno]
            code[cnt, 2:6] = dlist[idd, :]
            features[cnt, :] = [pzero, puno / 65536,
                                np.sqrt(pow((i - xd), 2) + pow((j - yd), 2)) / dd,
                                np.math.atan2((j - yd), (i - xd))]
            cnt += 1
    return PIFS(w, h, rsize, code, features)


def pifs_encode_tile(img, rsize, step, pr):
    #pr = list(permutations(range(0, 4)))
    #pr.reverse()
    n_cand = 20
    h = img.shape[0]
    w = img.shape[1]
    dd = np.sqrt(w * w + h * h)
    dsize = 2 * rsize
    dlist, features = pifs_extract_domains_features(img, dsize, step, pr)

    nrgs = len(range(0, h - rsize + 1, rsize)) * len(range(0, w - rsize + 1, rsize))
    fv = np.zeros((int(nrgs), int(rsize * rsize)), dtype='float32')
    idxnn = np.zeros((int(nrgs), n_cand), dtype='float32')

    cnt = 0
    for i in range(0, h - rsize + 1, rsize):
        for j in range(0, w - rsize + 1, rsize):
            r = img[i:(i + rsize), j:(j + rsize)]
            r = r.ravel()
            fv[cnt, :] = np.array((r - r.mean()) / r.std())
            cnt += 1
    for i in range(0, nrgs, 256):
        dist = cdist(fv[i: i + 256], features, 'cityblock')
        idxnn[i: i + 256, 0: n_cand] = np.argsort(dist, axis=1)[:, :n_cand]

    code = np.zeros((int(nrgs), 6), dtype='float32')
    features = np.zeros((int(nrgs), 4), dtype='float32')
    cnt = 0
    filt = np.ones((dsize, dsize), dtype='float32')+(1./dsize)
    for i in range(0, h - rsize + 1, rsize):
        for j in range(0, w - rsize + 1, rsize):
            for k in range(n_cand):
                idd = int(idxnn[cnt, k])
                iss = int(dlist[idd, 3])
                yd = int(dlist[idd, 0])
                xd = int(dlist[idd, 1])

                r = img[i:(i + rsize), j:(j + rsize)]
                d = img[yd:(yd + dsize), xd:(xd + dsize)]*filt
                d = pifs_permute_domain_patches(d, pr[iss])
                d = my_imresize(d)
                po = np.polyfit(d.ravel(), r.ravel(), 1)
                pzero = round(po[0] * 32) / 32
                puno = round(po[1] / 2) * 2
                if ((pzero <= 1.5 and pzero >= -1.5)):
                    code[cnt, 0:2] = [pzero, puno]
                    code[cnt, 2:6] = dlist[idd, :]
                    features[cnt, :] = [pzero, puno / 65536,
                                        np.sqrt(pow((i - xd), 2) + pow((j - yd), 2)) / dd,
                                        np.math.atan2((j - yd), (i - xd))]
                    break
                elif k == n_cand-1:
                    if r.mean() > 1.:
                        print('Experiment')
                        code[cnt, 0:2] = [0, r.mean()]
                    else:
                        print('MYExperiment')
                        code[cnt, 0:2] = [0, 1.]

                    code[cnt, 2:6] = dlist[idd, :]
                    features[cnt, :] = [pzero, puno / 65536,
                                        np.sqrt(pow((i - xd), 2) + pow((j - yd), 2)) / dd,
                                        np.math.atan2((j - yd), (i - xd))]

            cnt += 1

    return PIFS(w, h, rsize, code, features)


def pifs_decode_tile(pc: PIFS, niter, zoom, p):
    #p = list(permutations(range(0, 4)))
    #p.reverse()
    h = pc.h * zoom
    w = pc.w * zoom
    rsize = int(pc.rsize * zoom)
    b = np.zeros((int(h), int(w)), dtype='float32')
    dsize = 2 * rsize
    k = 0
    err = 1000
    b_old = b + 1.
    filt = np.ones((dsize, dsize), dtype='float32')+(1./dsize)
    while sum(sum(abs(b - b_old))) > 0:
        b_old = b.copy()
        cnt = 0
        for i in range(0, int(h - rsize) + 1, rsize):
            for j in range(0, int(w - rsize) + 1, rsize):
                a = pc.code[cnt, 0]
                bb = pc.code[cnt, 1]
                #yd = int((pc.code[cnt, 2] - 1) * zoom + 1)
                #xd = int((pc.code[cnt, 3] - 1) * zoom + 1)
                yd = int(pc.code[cnt, 2] * zoom)
                xd = int(pc.code[cnt, 3] * zoom)
                iss = int(pc.code[cnt, 5])

                d = b_old[yd: (yd + dsize), xd: (xd + dsize)]*filt
                #print(i,j, iss, pc.code[cnt, 2], pc.code[cnt, 3])
                d = pifs_permute_domain_patches(d, p[iss])
                d = my_imresize(d)
                r = np.clip(0., 255., (a * d) + bb)
                b[i: (i + rsize), j: (j + rsize)] = r.astype('float32')
                cnt = cnt + 1

        k = k + 1
        if k > 100:
            break

    #b = NormalizeData(b) * 65535.0;
    return b#.astype('uint16')


def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())


def generate_tiles_old(direc, n_img):
    opts: OPTS = hope_setup()
    w = opts.w
    h = opts.h
    tilesize = opts.tilesize
    range_size = opts.range_size
    domain_shift = opts.domain_shift
    num_dec_iter = opts.num_dec_iter
    # code_dir = 'Tiles_PIFSCodes/'
    # org_dir = 'Tiles_original/'
    # deco_dir = 'Tiles_decoded/'
    code_dir = 'Prova_res/'
    org_dir = 'Prova_res/'
    deco_dir = 'Prova_res/'
    pr = list(permutations(range(0, 4)))
    pr.reverse()

    for path in direc:
        dir = '/home/ascalella/dataset/Train/Original/' + path
        files = listdir(dir)
        n = 0
        if n_img == 0:
            n_img = len(files)
        for filepath, n in zip(files, range(n_img)):
            img = np.asarray(imread(dir + '/{0}'.format(filepath), plugin='tifffile'), dtype='uint16')
            tile = np.zeros((tilesize, tilesize, 3), dtype='uint16')
            for i in range(0, img.shape[0], tilesize):
                for j in range(0, img.shape[1], tilesize):
                    # Channel 0
                    img_t = img[i:i + tilesize, j:j + tilesize, 0]
                    R = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)
                    t = pifs_decode_tile(R, num_dec_iter, 1.0, pr)
                    tile[:, :, 0] = t
                    end_name = str(i) + '_' + str(j) + '_0'
                    tile_name = dir.replace(path, code_dir) + filepath.replace('.tif', '_' + end_name)
                    pkl.dump(R, open(tile_name + '.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
                    # Channel 1
                    img_t = img[i:i + tilesize, j:j + tilesize, 1]
                    G = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)
                    t = pifs_decode_tile(G, num_dec_iter, 1.0, pr)
                    tile[:, :, 1] = t
                    end_name = str(i) + '_' + str(j) + '_1'
                    tile_name = dir.replace(path, code_dir) + filepath.replace('.tif', '_' + end_name)
                    pkl.dump(G, open(tile_name + '.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
                    # Channel 2
                    img_t = img[i:i + tilesize, j:j + tilesize, 2]
                    B = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)
                    t = pifs_decode_tile(B, num_dec_iter, 1.0, pr)
                    tile[:, :, 2] = t
                    end_name = str(i) + '_' + str(j) + '_2'
                    tile_name = dir.replace(path, code_dir) + filepath.replace('.tif', '_' + end_name)
                    pkl.dump(B, open(tile_name + '.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
                    # Save original tile
                    tile_name = dir.replace(path, org_dir) + filepath.replace('.tif', '_' + end_name)
                    imsave(tile_name + '.tif', img[i:i + tilesize, j:j + tilesize, :], plugin='tifffile')
                    # Save decoded tile
                    tile_name = dir.replace(path, deco_dir) + filepath.replace('.tif', '_' + end_name)
                    imsave(tile_name + '.tif', tile, plugin='tifffile')
                    #pifs2bin([R, G, B], tile_name)

        print('DONE', filepath)


def generate_tiles_new(direc, n_img):
    opts: OPTS = hope_setup()
    w = opts.w
    h = opts.h
    tilesize = opts.tilesize
    range_size = opts.range_size
    domain_shift = opts.domain_shift
    num_dec_iter = opts.num_dec_iter
    code_dir = 'Tiles_PIFSCodes_Filter24/'
    org_dir = 'Tiles_original/'
    deco_dir = 'Tiles_decoded_Filter24/'
    code_dir_old = 'Tiles_PIFSCodes_OLD_Filter24/'
    #pr = np.asarray([[1,2,3,4],[1,3,2,4],[2,1,4,3],[2,4,1,3],[3,1,4,2],[3,4,1,2],[4,2,3,1],[4,3,2,1]], dtype='int8')-1
    pr = list(permutations(range(0, 4)))

    for path in direc:
        print(path)
        dir = '/home/ascalella/dataset/Train/' + path
        files = listdir(dir)
        n = 0
        result = []
        if n_img == 0:
            n_img = len(files)
        for filepath, n in zip(files, range(n_img)):
            proc = []
            img = np.clip(0., 255., np.asarray(imread(dir + '/{0}'.format(filepath), plugin='tifffile'), dtype='float32')/256.)
            pool = Pool(processes=11)
            for i in range(0, img.shape[0], tilesize):
                for j in range(0, img.shape[1], tilesize):
                    img_t = img[i:i + tilesize, j:j + tilesize, :]
                    end_name = '_' + str(i) + '_' + str(j)
                    midd_name = path.split('/')[-1]
                    bin_name = dir.replace(path, code_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
                    old_name = dir.replace(path, code_dir_old) + midd_name + '/' + filepath.replace('.tif', end_name)
                    tile_name = dir.replace(path, deco_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
                    
                    if i == img.shape[0]- tilesize & j == img.shape[1] - tilesize:
                        result.append(par_fun(img_t, tile_name, old_name, bin_name, range_size, domain_shift, pr, num_dec_iter))
                    else:
                        proc.append(pool.apply_async(par_fun, (img_t, tile_name, old_name, bin_name, range_size, domain_shift, pr, num_dec_iter)))
                    
            for i in proc:
                result.append(i.get())

            pool.close()
            pool.join()

            print('DONE', filepath)
        arr = np.asarray(result)
        print(path, 'PSNR: {0:.4f} - RMSE: {1:.4f}'.format(arr[:,0].mean(), arr[:,1].mean()))
        print(path, 'PSNRstd: {0:.4f} - RMSEstd: {1:.4f}'.format(arr[:,0].std(), arr[:,1].std()))


def par_fun(img_t, tile_name, old_name, bin_name, range_size, domain_shift, pr, num_dec_iter):
    a = 0
    b = 0
    tile = np.zeros(img_t.shape, dtype='float32')
    # Channel 0
    R = pifs_encode_tile(img_t[:,:,0].astype('float32'), range_size, domain_shift, pr)
    # Channel 1
    G = pifs_encode_tile(img_t[:,:,1].astype('float32'), range_size, domain_shift, pr)
    # Channel 2
    B = pifs_encode_tile(img_t[:,:,2].astype('float32'), range_size, domain_shift, pr)

    pkl.dump([R, G, B], open(old_name + '.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
       
    try:
        
        pifs2bin([R, G, B], bin_name, len(pr), 255)
        RGB = bin2pifs(bin_name + '.bin')

        R = RGB[0]
        G = RGB[1]
        B = RGB[2]

        t = pifs_decode_tile(R, num_dec_iter, 1.0, pr)
        tile[:, :, 0] = t

        t = pifs_decode_tile(G, num_dec_iter, 1.0, pr)
        tile[:, :, 1] = t

        t = pifs_decode_tile(B, num_dec_iter, 1.0, pr)
        tile[:, :, 2] = t

        # Save original tile
        #tile_name = dir.replace(path, org_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
        #imsave(tile_name + '.tif', img[i:i + tilesize, j:j + tilesize, :], plugin='tifffile')
        # Save decoded tile
        imsave(tile_name + '.tif', np.clip(0, 65535, tile*256).astype('uint16'), plugin='tifffile')
        a, b = Compute_PSNR(img_t, tile.astype('float32'))

    except Exception as e:
        print('Problem with: ', tile_name, '\n',e)
    
    return [a, b]


def generate_tiles(direc, n_img):
    opts: OPTS = hope_setup()
    w = opts.w
    h = opts.h
    tilesize = opts.tilesize
    range_size = opts.range_size
    domain_shift = opts.domain_shift
    num_dec_iter = opts.num_dec_iter
    #code_dir = 'Tiles_PIFSCodes/'
    org_dir = 'Tiles_original/'
    #deco_dir = 'Tiles_decoded/'
    #code_dir_old = 'Tiles_PIFSCodes_OLD/'
    #pr = list(permutations(range(0, 4)))
    pr = np.asarray([[1,2,3,4],[1,3,2,4],[2,1,4,3],[2,4,1,3],[3,1,4,2],[3,4,1,2],[4,2,3,1],[4,3,2,1]], dtype='int8')-1
    
    code_dir = 'Prova/'
    deco_dir = 'Prova/'
    code_dir_old = 'Prova/'
    

    for path in direc:
        dir = '/home/ascalella/dataset/Train/' + path
        files = listdir(dir)
        n = 0
        psnr = 0
        rmse = 0
        if n_img == 0:
            n_img = len(files)
        for filepath, n in zip(files, range(n_img)):
            img = np.clip(0., 255., np.asarray(imread(dir + '/{0}'.format(filepath), plugin='tifffile'), dtype='float32')/256.)            
            tile = np.zeros((tilesize, tilesize, 3), dtype='float32')
            for i in range(0, img.shape[0], tilesize):
                for j in range(0, img.shape[1], tilesize):
                    # Channel 0
                    img_t = img[i:i + tilesize, j:j + tilesize, 0]
                    R = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)
                    # Channel 1
                    img_t = img[i:i + tilesize, j:j + tilesize, 1]
                    G = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)
                    # Channel 2
                    img_t = img[i:i + tilesize, j:j + tilesize, 2]
                    B = pifs_encode_tile(img_t.astype('float32'), range_size, domain_shift, pr)

                    end_name = '_' + str(i) + '_' + str(j)
                    midd_name = path.split('/')[-1]
                    bin_name = dir.replace(path, code_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
                    tile_name = dir.replace(path, code_dir_old) + midd_name + '/' + filepath.replace('.tif', end_name)
                    pkl.dump([R, G, B], open(tile_name + '.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
                    
                    #try:
                        
                    pifs2bin([R, G, B], bin_name, len(pr), 255.)
                    RGB = bin2pifs(bin_name + '.bin')

                    R = RGB[0]
                    G = RGB[1]
                    B = RGB[2]
                    

                    t = pifs_decode_tile(R, num_dec_iter, 1.0, pr)
                    tile[:, :, 0] = t

                    t = pifs_decode_tile(G, num_dec_iter, 1.0, pr)
                    tile[:, :, 1] = t

                    t = pifs_decode_tile(B, num_dec_iter, 1.0, pr)
                    tile[:, :, 2] = t

                    # Save original tile
                    #tile_name = dir.replace(path, org_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
                    #imsave(tile_name + '.tif', img[i:i + tilesize, j:j + tilesize, :], plugin='tifffile')
                    # Save decoded tile
                    tile_name = dir.replace(path, deco_dir) + midd_name + '/' + filepath.replace('.tif', end_name)
                    imsave(tile_name + '.tif', np.clip(0, 65535, tile*256).astype('uint16'), plugin='tifffile')
                    a, b = Compute_PSNR(img[i:i + tilesize, j:j + tilesize, :], tile)

                    psnr += a
                    rmse += b

                    #except:
                        #print('Problem with: ', filepath)
            print('DONE', filepath)
        print(path, 'PSNR: {0:.4f} - RMSE: {1:.4f}'.format(psnr/(n_img*12), rmse/(n_img*12)))


def pifs2bin(PIFSch, out_dir, n_perm, r_col):
    pi: PIFS = PIFSch[0]
    im_sz = log(pi.w, 2)
    rsize = log(pi.rsize, 2)
    n_p = 0
    n_c = 0
    n_range = pi.code.shape[0]
    if n_perm > 8:
        n_p = 1
        last = 'u5'
    else:
        last = 'u3'

    if r_col > 256.:
        b_beta = 'u13'
        n_c = 1
        dif_b = 2**15
        shift = 3
        tipe = 'uint16'
        fact = 8.

    else:
        b_beta = 'u6'
        dif_b = 2**7
        shift = 2
        tipe = 'uint8'
        fact = 2.



    out = bytearray(round((8 + 3 * calcsize('u5'+ b_beta +'u6u6'+last)*n_range) / 8.) +1)
    off = 0
    pack_into('u4u3u1u1', out, off, im_sz, rsize, n_p, n_c, fill_padding=False)
    off += 9

    for pi in PIFSch:
        pack_into('u5'*n_range, out, off, *np.rint((pi.code[:, 0] + 1.5) * 10.))
        off += calcsize('u5'*n_range)
        pack_into(b_beta*n_range, out, off, *((np.rint(pi.code[:, 1]/ fact + dif_b)).astype(tipe) >> shift))
        off += calcsize(b_beta*n_range)
        pack_into('u6'*n_range, out, off, *(pi.code[:, 2] / 8))
        off += calcsize('u6'*n_range)
        pack_into('u6'*n_range, out, off, *(pi.code[:, 3] / 8))
        off += calcsize('u6'*n_range)
        pack_into(last*n_range, out, off, *pi.code[:, 5])
        off += calcsize(last*n_range)

    with open(out_dir + '.bin', 'wb') as file:
        file.write(out)


def bin2pifs(path_bin):
    PIFSch = []
    with open(path_bin, 'rb') as file:
        total = file.read()
    a, b, c, d = unpack_from('u4u3u1u1', total, 0)
    im_sz = 2 ** a
    rsize = 2 ** b
    nrange = int(im_sz / rsize) ** 2
    off = 9
    #print('ECCO ',a,b,c,d, nrange)

    if c==1:
        last = 'u5'
    else:
        last = 'u3'

    if d==1:
        b_beta = 'u13'
        dif_b = 2**15
        shift = 3
        fact = 8.
    else:
        b_beta = 'u6'
        dif_b = 2**7
        shift = 2
        fact = 2.



    for i in range(3):
        code = np.zeros((nrange, 6), dtype='float32')
        alpha = np.asarray(list(unpack_from('u5'*nrange, total, off)))
        off += calcsize('u5'*nrange)
        code[:, 0] = (alpha / 10.) - 1.5
        beta = np.asarray(list(unpack_from(b_beta*nrange, total, off)))
        off += calcsize(b_beta*nrange)
        code[:, 1] = ((beta << shift) - dif_b) * 2.
        xd = np.asarray(list(unpack_from('u6'*nrange, total, off)))
        off += calcsize('u6'*nrange)
        code[:, 2] = xd * 8.
        yd = np.asarray(list(unpack_from('u6'*nrange, total, off)))
        off += calcsize('u6'*nrange)
        code[:, 3] = yd * 8.
        code[:, 4] = rsize * 2
        ism = np.asarray(list(unpack_from(last*nrange, total, off)))
        off += calcsize(last*nrange)
        code[:, 5] = ism
        PIFSch.append(PIFS(im_sz, im_sz, rsize, code, []))

    return PIFSch


def Compute_PSNR(a, b):

    a= a.astype('float32')
    b= b.astype('float32')

    b[:,:,0] = b[:,:,0] - b[:,:,0].mean() + a[:,:,0].mean()
    b[:,:,1] = b[:,:,1] - b[:,:,1].mean() + a[:,:,1].mean()
    b[:,:,2] = b[:,:,2] - b[:,:,2].mean() + a[:,:,2].mean()
 
    rmse= np.sqrt(np.mean((a-b)**2))
    psnr=20*np.log10(a.max()/rmse)

    return psnr, rmse 


def oldpifs2bin(PIFSch, out_dir, n_perm):
    pi: PIFS = PIFSch[0]
    im_sz = log(pi.w, 2)
    rsize = log(pi.rsize, 2)
    n_p = 0

    if n_perm >8:
        fmt = 'u5u13u6u6u5'
        n_p = 1
    else:
        fmt = 'u5u13u6u6u3'

    out = bytearray(round((8 + 3 * pi.code.shape[0] * calcsize(fmt)) / 8.))
    off = 0
    pack_into('u4u3u1', out, off, im_sz, rsize, n_p, fill_padding=False)
    off += 8

    for i in range(3):
        pi = PIFSch[i]
        for j in range(pi.code.shape[0]):
            pack_into(fmt, out, off, round((pi.code[j, 0] + 1.5) * 10.), np.uint16((pi.code[j, 1] / 8.) + 2**15) >> 3,
                      pi.code[j, 2] / 8, pi.code[j, 3] / 8, pi.code[j, 5], fill_padding=False)
            off += calcsize(fmt)

    with open(out_dir + '.bin', 'wb') as file:
        file.write(out)


def oldbin2pifs(path_bin):
    PIFSch = []
    with open(path_bin, 'rb') as file:
        total = file.read()
    a, b, c = unpack_from('u4u3u1', total, 0)
    im_sz = 2 ** a
    rsize = 2 ** b
    nrange = int(im_sz / rsize) ** 2
    off = 8

    if c==1:
        fmt = 'u5u13u6u6u5'
    else:
        fmt = 'u5u13u6u6u3'


    for i in range(3):
        code = np.zeros((nrange, 6), dtype='float32')
        for j in range(nrange):
            flag = unpack_from(fmt, total, off)
            code[j, 0] = (flag[0] / 10.) - 1.5
            code[j, 1] = ((int(flag[1]) << 3) - 2**15) * 8.
            code[j, 2] = flag[2] * 8.
            code[j, 3] = flag[3] * 8.
            code[j, 4] = rsize * 2
            code[j, 5] = flag[4]
            off += calcsize(fmt)
        PIFSch.append(PIFS(im_sz, im_sz, rsize, code, []))

    return PIFSch

