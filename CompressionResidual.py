import numpy as np
import RLE as rle
import bitarray as bt
import time
import pickle as pkl
import matplotlib.pyplot as plt
import glymur as GL
from PIL import Image
from bitstruct import *
from collections import Counter
from scipy.fftpack import dct, idct
from bitarray.util import huffman_code
from os import stat, listdir
from skimage.io import imread
from matplotlib import use
from skimage.metrics import structural_similarity as ssim
#from tensorflow.image import image_gradients
#from tensorflow import convert_to_tensor



def rgb2gray(rgb):

	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

	return gray


def standardize(a,b):
	a= a.astype('float32')
	b= b.astype('float32')

	b[:,:,0] = (((b[:,:,0] - b[:,:,0].mean())/b[:,:,0].std()) * a[:,:,0].std()) + a[:,:,0].mean()
	b[:,:,1] = (((b[:,:,1] - b[:,:,1].mean())/b[:,:,1].std()) * a[:,:,1].std()) + a[:,:,1].mean()
	b[:,:,2] = (((b[:,:,2] - b[:,:,2].mean())/b[:,:,2].std()) * a[:,:,2].std()) + a[:,:,2].mean()

	return b.clip(0, a.max())

def standardize2(a,b):
	a= a.astype('float32')
	b= b.astype('float32')

	b[:,:,0] = b[:,:,0] - b[:,:,0].mean() + a[:,:,0].mean()
	b[:,:,1] = b[:,:,1] - b[:,:,1].mean() + a[:,:,1].mean()
	b[:,:,2] = b[:,:,2] - b[:,:,2].mean() + a[:,:,2].mean()

	return b.clip(0, a.max())


def dct2(a):
	return dct(dct(a.transpose(), norm='ortho').transpose(), norm='ortho')


def idct2(a):
	return idct(idct(a.transpose(), norm='ortho').transpose(), norm='ortho')


def zigzag(input):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    i = 0

    output = np.zeros(( vmax * hmax))

    while ((v < vmax) and (h < hmax)):
        
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax -1):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[i] = input[v, h]
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                output[i] = input[v, h]

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[i] = input[v, h]
            break

    return output


def inverse_zigzag(input, vmax, hmax):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    output = np.zeros((vmax, hmax))
    i = 0

    while ((v < vmax) and (h < hmax)): 
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax - 1):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[v, h] = input[i]
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                output[v, h] = input[i]
                if (v == vmax):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[v, h] = input[i]
            break

    return output


def MyDct(img, n_coeff, blksize, quality):

	if len(img.shape) > 2:
		img = rgb2gray(img)

	dctcoef = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
	for i in range(0, img.shape[0], blksize):
		for j in range(0, img.shape[1], blksize):
			dctcoef[i: i+blksize, j: j+blksize] = dct2(img[i: i+blksize, j: j+blksize])

	if n_coeff == 36:
		filt = np.array([[1, 1, 1, 1, 1, 1, 1, 1],				
						[1, 1, 1, 1, 1, 1, 1, 0],
						[1, 1, 1, 1, 1, 1, 0, 0],
						[1, 1, 1, 1, 1, 0, 0, 0],
						[1, 1, 1, 1, 0, 0, 0, 0],
						[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 28:
		filt = np.array([[1, 1, 1, 1, 1, 1, 1, 0],
						[1, 1, 1, 1, 1, 1, 0, 0],
						[1, 1, 1, 1, 1, 0, 0, 0],
						[1, 1, 1, 1, 0, 0, 0, 0],
						[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 21:
		filt = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
						[1, 1, 1, 1, 1, 0, 0, 0],
						[1, 1, 1, 1, 0, 0, 0, 0],
						[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 15:
		filt = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
						[1, 1, 1, 1, 0, 0, 0, 0],
						[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 10:	
		filt = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
						[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 6:
		filt = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	elif n_coeff == 3:
		filt = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]])
	else:
		filt = np.ones((8,8))

	quantization_value = np.array([[16,  11,  10,  16,  24,   40,   51,   61],
									 [12,  12,  14,  19,  26,   58,   60,   55],
									 [14,  13,  16,  24,  40,   57,   69,   56],
									 [14,  17,  22,  29,  51,   87,   80,   62],
									 [18,  22,  37,  56,  68,   109,  103,  77],
									 [24,  35,  55,  64,  81,   104,  113,  92],
									 [49,  64,  78,  87,  103,  121,  120,  101],
									 [72,  92,  95,  98,  112,  100,  103,  99]], dtype='float32')


	if quality < 50:
		s = 5000./quality
	elif quality >= 50 and quality < 100:
		s = 200. - 2.*quality
	else:
		s = 1.

	q_mtx = (s*quantization_value + 50.)/100.


	if blksize > 8:
		blkf_mtx = np.zeros((blksize, blksize), dtype='float32')
		blkf_mtx[:8, :8] = filt.astype('float32')
		blkq_mtx = np.ones((blksize, blksize), dtype='float32')
		blkq_mtx[:8, :8] = q_mtx
	else:
		blkf_mtx = filt.astype('float32')
		blkq_mtx = q_mtx

	cuttcoef = np.ones((img.shape[0], img.shape[1]), dtype='int32')
	for i in range(0, dctcoef.shape[0], blksize):
		for j in range(0, dctcoef.shape[1], blksize):
			cuttcoef[i: i+blksize, j: j+blksize] = np.rint((dctcoef[i: i+blksize, j: j+blksize] * blkf_mtx) / blkq_mtx)

	return cuttcoef


def MyIdct(cuttcoef, blksize, quality):

	quantization_value = np.array([[16,  11,  10,  16,  24,   40,   51,   61],
									 [12,  12,  14,  19,  26,   58,   60,   55],
									 [14,  13,  16,  24,  40,   57,   69,   56],
									 [14,  17,  22,  29,  51,   87,   80,   62],
									 [18,  22,  37,  56,  68,   109,  103,  77],
									 [24,  35,  55,  64,  81,   104,  113,  92],
									 [49,  64,  78,  87,  103,  121,  120,  101],
									 [72,  92,  95,  98,  112,  100,  103,  99]], dtype='float32')


	if quality < 50.:
		s = 5000./quality
	elif quality >= 50. and quality < 100.:
		s = 200. - 2.*quality
	else:
		s = 1.

	q_mtx = (s*quantization_value + 50.)/100.

	if blksize > 8:
		blkq_mtx = np.ones((blksize, blksize), dtype='float32')
		blkq_mtx[:8, :8] = q_mtx
	else:
		blkq_mtx = q_mtx

	img = np.zeros((cuttcoef.shape[0], cuttcoef.shape[1]), dtype='float32')
	for i in range(0, cuttcoef.shape[0],blksize):
		for j in range(0, cuttcoef.shape[1],blksize):
			img[i: i+blksize, j: j+blksize] = idct2(cuttcoef[i: i+blksize, j: j+blksize] * blkq_mtx)


	return img


def MyCompression(img, n_coeff, blksize, quality, lossless, out_dir):

	img = img.astype('float32')

	if np.max(img) > 255.:
		img = img/256.

	if quality > 10:
		quality = 10.
	elif quality <=0:
		quality = 1.

	im_sz = img.shape[0]
	
	if lossless:
		out = bytearray(round((1 + 16 + im_sz*im_sz*3*8) / 8.) + 1)
		off = 0
		pack_into('b1u16', out, off, lossless, im_sz, fill_padding=False)
		off += 17
		img = img.clip(0, 255).astype('uint8')
		Rout = img[:, :, 0]
		Gout = img[:, :, 1]
		Bout = img[:, :, 2]
		pack_into('r'+str(im_sz*im_sz*8)+'r'+str(im_sz*im_sz*8)+'r'+str(im_sz*im_sz*8), out, off, bytearray(Rout.ravel()), bytearray(Gout.ravel()), bytearray(Bout.ravel()),fill_padding=False)

	else:
		R = MyDct(img[:, :, 0], n_coeff, blksize, quality*10)
		G = MyDct(img[:, :, 1], n_coeff, blksize, quality*10)
		B = MyDct(img[:, :, 2], n_coeff, blksize, quality*10)

		Rz = np.zeros((R.shape[0]*R.shape[1]), dtype='int32')
		Gz = np.zeros((G.shape[0]*G.shape[1]), dtype='int32')
		Bz = np.zeros((B.shape[0]*B.shape[1]), dtype='int32')

		x = 0
		for i in range(0, R.shape[0],blksize):
			for j in range(0, R.shape[1],blksize):
				Rz[x:x + blksize*blksize] = zigzag(R[i:i + blksize, j:j + blksize])
				Gz[x:x + blksize*blksize] = zigzag(G[i:i + blksize, j:j + blksize])
				Bz[x:x + blksize*blksize] = zigzag(B[i:i + blksize, j:j + blksize])
				x += blksize*blksize
		
		Rin = np.asarray(rle.encode(Rz), dtype='int32')
		Gin = np.asarray(rle.encode(Gz), dtype='int32')
		Bin = np.asarray(rle.encode(Bz), dtype='int32')

		RGB = np.concatenate((Rin.ravel(), Gin.ravel(), Bin.ravel()))

		dic = huffman_code(Counter(RGB.ravel()))
		codeR = bt.bitarray()
		codeR.encode(dic, Rin.ravel())
		codeG = bt.bitarray()
		codeG.encode(dic, Gin.ravel())
		codeB = bt.bitarray()
		codeB.encode(dic, Bin.ravel())

		totLenVal = 0
		for i in dic.keys(): totLenVal += len(dic[i])

		key = list(dic.keys())
		value = list(dic.values())

		a = int(np.max(key))
		b = int(np.min(key))
		c = int(np.max([a, np.abs(b)]))
		n_bk = c.bit_length() + 1


		off = 0
		fmt = 'b1u8u4u16u32u32u32u16u5'
		out = bytearray(round((calcsize(fmt) + len(codeR) + len(codeG) + len(codeB) + len(dic)*(n_bk + 5) + totLenVal) / 8.)+1)
		pack_into(fmt, out, off, lossless, blksize, quality, im_sz, len(codeR), len(codeG), len(codeB),len(dic), n_bk,fill_padding=False)
		off += calcsize(fmt)

		for i in range(len(dic)):
			fmt = 'u5s'+str(n_bk)+'r'+str(len(value[i]))
			pack_into(fmt, out, off, len(value[i]), key[i], value[i], fill_padding=False)
			off += calcsize(fmt)

		pack_into('r'+str(len(codeR))+'r'+str(len(codeG))+'r'+str(len(codeB)), out, off, codeR, codeG, codeB, fill_padding=False)

	with open(out_dir + '.bin', 'wb') as file:
			file.write(out)

	return out


def MyDecompression(path_bin):

	with open(path_bin, 'rb') as file:
		raw = file.read()
	
	off = 0
	lossless = unpack_from('b1', raw, off)
	off += 1
	if lossless[0] == 1:
		size = unpack_from('u16', raw, off)
		im_sz = size[0]
		off += 16
		Rb, Gb, Bb = unpack_from('r'+str(im_sz*im_sz*8)+'r'+str(im_sz*im_sz*8)+'r'+str(im_sz*im_sz*8), raw, off)
		R = np.array(list(Rb), dtype='uint16').reshape((im_sz, im_sz))
		G = np.array(list(Gb), dtype='uint16').reshape((im_sz, im_sz))
		B = np.array(list(Bb), dtype='uint16').reshape((im_sz, im_sz))

	else:

		blksize, quality, im_sz, lenR, lenG, lenB, len_dic, n_bk = unpack_from('u8u4u16u32u32u32u16u5', raw, off)
		if quality == 0:
			quality = 5
		off += calcsize('u8u4u16u32u32u32u16u5')
		dic ={}
		for i in range(len_dic):
			len_k, val = unpack_from('u5s'+str(n_bk), raw, off)
			off += calcsize('u5s'+str(n_bk))
			key = unpack_from('r'+str(len_k), raw, off)
			off += calcsize('r'+str(len_k))
			k = bt.bitarray()
			k.frombytes(bytes(key[0]))
			dic[val] = k[:len_k]

		Rr, Gr, Br = unpack_from('r'+str(lenR)+'r'+str(lenG)+'r'+str(lenB),raw, off)

		tree = bt.decodetree(dic)
		Rb = bt.bitarray()
		Gb = bt.bitarray()
		Bb = bt.bitarray()

		Rb.frombytes(bytes(Rr))
		Gb.frombytes(bytes(Gr))
		Bb.frombytes(bytes(Br))

		Rh = Rb[:lenR].decode(tree)
		Gh = Gb[:lenG].decode(tree)
		Bh = Bb[:lenB].decode(tree)

		Rh = np.reshape(Rh, (2, int(len(Rh)/2)))
		Gh = np.reshape(Gh, (2, int(len(Gh)/2)))
		Bh = np.reshape(Bh, (2, int(len(Bh)/2)))

		Rr = rle.decode(Rh[0,:], Rh[1,:])
		Gr = rle.decode(Gh[0,:], Gh[1,:])
		Br = rle.decode(Bh[0,:], Bh[1,:])

		Rz = np.zeros((im_sz, im_sz), dtype='float32')
		Gz = np.zeros((im_sz, im_sz), dtype='float32')
		Bz = np.zeros((im_sz, im_sz), dtype='float32')

		x = 0
		for i in range(0, im_sz, blksize):
			for j in range(0, im_sz, blksize):
				Rz[i:i + blksize,j:j + blksize] = inverse_zigzag(Rr[x:x + blksize*blksize], blksize, blksize)
				Gz[i:i + blksize,j:j + blksize] = inverse_zigzag(Gr[x:x + blksize*blksize], blksize, blksize)
				Bz[i:i + blksize,j:j + blksize] = inverse_zigzag(Br[x:x + blksize*blksize], blksize, blksize)
				x += blksize*blksize

		R = MyIdct(Rz, blksize, quality*10)
		G = MyIdct(Gz, blksize, quality*10)
		B = MyIdct(Bz, blksize, quality*10)


	out = np.zeros((im_sz,im_sz,3), dtype='float32')
	out[:, :, 0] = R
	out[:, :, 1] = G
	out[:, :, 2] = B

	return out


def rec_mask(img, blk_sz, qt, tresh):
	ind = int(blk_sz/2)
	if blk_sz == 8:
		tree = '0'
		#dctcoef = zigzag(dct2(img))
		dctcoef = zigzag(MyDct(img, 64, blk_sz, qt))
		
	else:
		if blk_sz > 64 or np.linalg.norm(img) > tresh:
			half = int(blk_sz/2)
			tree = '1'
			dctcoef_l = []
			a , b = rec_mask(img[:ind, :ind], ind, qt, tresh)
			tree += a
			dctcoef_l += list(b)
			a , b = rec_mask(img[:ind, ind:], ind, qt, tresh)
			tree += a
			dctcoef_l += list(b)
			a , b = rec_mask(img[ind:, :ind], ind, qt, tresh)
			tree += a
			dctcoef_l += list(b)
			a , b = rec_mask(img[ind:, ind:], ind, qt, tresh)
			tree += a
			dctcoef_l += list(b)
			dctcoef = np.asarray(dctcoef_l)
		else:
			tree = '0'
			dctcoef = zigzag(MyDct(img, 64, blk_sz, qt))
			#dctcoef = zigzag(dct2(img))

	return tree, dctcoef

def Irec_mask(tree, dctcoef, blk_sz, qt):
	if tree[0] == '1':
		half = int(blk_sz/2)
		img = np.zeros((blk_sz, blk_sz))
		i, skip, n_c = Irec_mask(tree[1:], dctcoef, half, qt)
		img[:half, :half] = i
		n_ca = n_c
		pos_tree = skip +1
		i, skip, n_c = Irec_mask(tree[pos_tree:], dctcoef[n_ca*64:], half, qt)
		img[:half, half:] = i
		n_ca += n_c
		pos_tree += skip
		i, skip, n_c = Irec_mask(tree[pos_tree:], dctcoef[n_ca*64:], half, qt)
		img[half:, :half] = i
		n_ca += n_c
		pos_tree += skip
		i, skip, n_c = Irec_mask(tree[pos_tree:], dctcoef[n_ca*64:], half, qt)
		img[half:, half:] = i
		n_ca += n_c
		pos_tree += skip
	else:
		dco = np.zeros((blk_sz * blk_sz))
		dco[:64] = dctcoef[:64]
		idco = inverse_zigzag(dco, blk_sz, blk_sz)
		img = idct2(idco)
		#img = MyIdct(idco, blk_sz, qt)
		n_ca = 1
		pos_tree = 1
	
	return img, pos_tree, n_ca


def Compute_PSNR(a, b):

	a= a.astype('float32')
	b= b.astype('float32')

	#b[:,:,0] = b[:,:,0] - b[:,:,0].mean() + a[:,:,0].mean()
	#b[:,:,1] = b[:,:,1] - b[:,:,1].mean() + a[:,:,1].mean()
	#b[:,:,2] = b[:,:,2] - b[:,:,2].mean() + a[:,:,2].mean()
 
	rmse= np.sqrt(np.mean((a-b)**2))
	psnr=20*np.log10(a.max()/rmse)

	return psnr, rmse 

#n_coeff = [3,6,10,15,21,28,36,64]
def comprTest(img_path, blk_sz = [8, 16, 32, 64] , n_coeff = [64], quality = [0.5,1,2,3,4,5,6,7,8,9,10]):

	d = img_path.rsplit('/')
	dim_org = stat(img_path).st_size
	upp = img_path.replace(d[-2]+ '/' +d[-1], '')
	#pifs = upp + 'PIFSbin/' + d[-1].replace('.tif', '.bin')
	dim_pifs = 10754#9986 #10754-11138-10370-13441 stat(pifs).st_size
	original = np.array(imread(img_path, plugin='tifffile'), dtype='float32')
	decoded = np.array(imread(upp+'Tiles_net_MYmean24/'+d[-1], plugin='tifffile'), dtype='float32')
	dif = original - decoded #.clip(0, original.max())

	strinf_f = ''
	psnr, rmse = Compute_PSNR(original, decoded)
	ss = ssim(original/65535., decoded/65535., data_range=1.0, multichannel=True)
	wpsnr = wPSNR(original, decoded)


	strinf_f = strinf_f + 'PSNR(Original, Decoded) = {0:.4f} \n'.format(psnr)
	strinf_f = strinf_f + 'WPSNR(Original, Decoded) = {0:.4f} \n'.format(wpsnr)
	strinf_f = strinf_f + 'RMSE(Original, Decoded) = {0:.4f} \n'.format(rmse)
	strinf_f = strinf_f + 'SSMI(Original, Decoded) = {0:.4f} \n\n'.format(ss)

	psnr, rmse = Compute_PSNR(original, decoded + np.rint((dif/256.)).astype('float32')*256.)
	ss = ssim(original/65535., (decoded + np.rint((dif/256.)).astype('float32')*256.)/65535., data_range=1.0, multichannel=True)
	wpsnr = wPSNR(original, decoded + np.rint((dif/256.)).astype('float32')*256.)


	strinf_f = strinf_f + 'PSNR(Original, Decoded + double(uint8((Original - Decoded)/256.0)*256)) = {0:.4f} \n'.format(psnr)
	strinf_f = strinf_f + 'WPSNR(Original, Decoded + double(uint8((Original - Decoded)/256.0)*256)) = {0:.4f} \n'.format(wpsnr)
	strinf_f = strinf_f + 'RMSE(Original, Decoded + double(uint8((Original - Decoded)/256.0)*256)) = {0:.4f} \n'.format(rmse)
	strinf_f = strinf_f + 'SSMI(Original, Decoded + double(uint8((Original - Decoded)/256.0)*256)) = {0:.4f} \n'.format(ss)
	resQT = []
	for qt in quality:
		ps = np.empty((len(blk_sz), len(n_coeff)))
		wps = np.empty((len(blk_sz), len(n_coeff)))
		rm = np.empty((len(blk_sz), len(n_coeff)))
		ssi = np.empty((len(blk_sz), len(n_coeff)))
		co_t = np.empty((len(blk_sz), len(n_coeff)))
		de_t = np.empty((len(blk_sz), len(n_coeff)))
		dim = np.empty((len(blk_sz), len(n_coeff)))
		ratio = np.empty((len(blk_sz), len(n_coeff)))
		i = 0
		for blk in blk_sz:
			j = 0
			for coeff in n_coeff:
				out_dir = upp + 'Residual/' + d[-1].replace('.tif','') + '_{0}_{1}_{2}_'.format(int(qt), blk, coeff)
				tic = time.time()
				out = MyCompression(dif, coeff, blk, qt, 0, out_dir)
				comp_time = time.time() - tic
				tic = time.time()
				dif_dec = MyDecompression(out_dir + '.bin')*256.
				deco_time = time.time() - tic;


				
				dim_diff = stat(out_dir + '.bin').st_size
				
				psnr, rmse = Compute_PSNR(original, standardize2(original, decoded + dif_dec))
				a, b = Compute_PSNR(original, decoded + dif_dec)
				
				if a > psnr:
					psnr = a
					rmse = b
				
				ss = ssim(original/65535., standardize2(original, decoded + dif_dec)/65535., data_range=1.0, multichannel=True)
				a = ssim(original/65535., (decoded + dif_dec)/65535., data_range=1.0, multichannel=True)

				if a > ss:
					ss = a

				wpsnr = wPSNR(original, standardize2(original, decoded + dif_dec))
				a = wPSNR(original, decoded + dif_dec)

				if a > wpsnr:
					wpsnr = a
				
				strinf_f = strinf_f + '\nQuality: {0}\t Block size: {1} \t N.Coeff: {2}\n'.format(qt, blk, coeff)
				strinf_f = strinf_f + 'PSNR(Original, Decoded + Diff) = {0:.4f} \n'.format(psnr)
				strinf_f = strinf_f + 'RMSE(Original, Decoded + Diff) = {0:.4f} \n'.format(rmse)
				strinf_f = strinf_f + 'SSMI(Original, Decoded + Diff) = {0:.4f} \n'.format(ss)
				strinf_f = strinf_f + 'Dimension(Diff_Comp) = {0:.4f} bytes - {1:.4f} Kbytes\n'.format(dim_diff, dim_diff/1024.)
				strinf_f = strinf_f + 'CompressionRatio(Original, Decoded + Diff_Comp) = {0:.4f} \n'.format(dim_org/(dim_diff+dim_pifs))
				strinf_f = strinf_f + 'Compression elapsed time: {0:.4f} sec\nDecompression elapsed time: {1:.4f} sec\n'.format(comp_time, deco_time)
				ps[i,j] = psnr
				wps[i,j] = wpsnr
				rm[i,j] = rmse
				ssi[i,j] = ss
				co_t[i,j] = comp_time
				de_t[i,j] = deco_time
				dim[i,j] = dim_diff
				ratio[i,j] = dim_org/(dim_diff+dim_pifs)
				j += 1
			i += 1
		
		result = Summary(blk_sz, n_coeff, quality, ps, rm, co_t, de_t, dim, ratio, ssi, wps)
		resQT.append(result)	


	with open(upp + 'Report/' + d[-1].replace('.tif', '_report.txt'), 'wt') as file:
		file.write(strinf_f)

	pkl.dump(resQT, open('/home/ascalella/dataset/Test/Result/' + d[-1].replace('.tif','.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)


	return resQT


class Summary():
	def __init__(self, blksize, n_coeff, quality, psnr, rmse, com_t, deco_t, dim_res, ratio, ssmi, wpsnr):
		self.blksize = blksize
		self.n_coeff = n_coeff
		self.quality = quality
		self.psnr = psnr
		self.rmse = rmse
		self.com_t = com_t
		self.deco_t = deco_t
		self.dim_res = dim_res
		self.ratio = ratio
		self.ssmi = ssmi
		self.wpsnr = wpsnr


class Summary_JPEG():
	def __init__(self, quality, psnr, rmse, ratio, ssmi, wpsnr):
		self.quality = quality
		self.psnr = psnr
		self.rmse = rmse
		self.ratio = ratio
		self.ssmi = ssmi
		self.wpsnr = wpsnr


def plot_res(path = None, out_dir = None):
	use('Agg')
	if path is None: path = '/home/ascalella/dataset/Test/Result'
	if out_dir is None: out_dir = '/home/ascalella/dataset/Test/Images'

	file = listdir(path)
	step = 0.5
	for i in range(10):
		psnr = np.zeros((4,8))
		ratio = np.zeros((4,8))
		for fi in file:
			res = pkl.load( open( path + '/' + fi,'rb' ))
			psnr += res[i].psnr
			ratio += res[i].ratio
		plt.xlabel('Compression Ratio', fontsize=16, fontweight='bold')
		plt.ylabel('PSNR', fontsize=16, fontweight='bold')
		plt.xticks(np.arange(0 ,130, step=5))
		if i == 9:
			 step = 1.
		plt.yticks(np.arange(int(((psnr)/len(file)).min()) - 5 , int(((psnr)/len(file)).max()) + 5, step=step))
		plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [PSNR: Original, DecodedNet + ResidualComp] \n Quality: {0}'.format(i + 1), fontweight='bold')
		plt.grid(True)
		plt.plot(ratio[0, :]/len(file), psnr[0,:]/len(file), 'gs-', label='Block Size: 8')
		plt.plot(ratio[1, :]/len(file), psnr[1,:]/len(file), 'bo-', label='Block Size: 16')
		plt.plot(ratio[2, :]/len(file), psnr[2,:]/len(file), 'rd-', label='Block Size: 32')
		plt.plot(ratio[3, :]/len(file), psnr[3,:]/len(file), 'k*-', label='Block Size: 64')
		plt.legend()
		figure = plt.gcf()
		figure.set_size_inches(12, 6)
		plt.savefig(out_dir + '/QualityPlot' + str(i+1) + '.png')
		plt.close()


def plot_res2(path = None, out_dir = None):
	use('Agg')
	if path is None: path = '/home/ascalella/dataset/Test/Result'
	if out_dir is None: out_dir = '/home/ascalella/dataset/Test/Images'
	d = path.rsplit('/')

	file = listdir(path)
	step = 0.5
	
	psnr = np.zeros((4,11))
	wpsnr = np.zeros((4,11))
	ratio = np.zeros((4,11))
	similiar = np.zeros((4,11))
	psnr_jpg = np.zeros((13))
	wpsnr_jpg = np.zeros((13))
	ratio_jpg = np.zeros((13))
	similiar_jpg = np.zeros((13))
	psnr_jpg2000 = np.zeros((14))
	wpsnr_jpg2000 = np.zeros((14))
	ratio_jpg2000 = np.zeros((14))
	similiar_jpg2000 = np.zeros((14))
	res_jpg = pkl.load( open( '/home/ascalella/dataset/Test/Result_JPEG/TOTALE_END3.pkl','rb' ))
	res_jpg2000 = pkl.load( open( '/home/ascalella/dataset/Test/Result_JPEG2000/TOTALE3.pkl','rb' ))
	res_jpg2000mix8 = pkl.load( open( '/home/ascalella/dataset/Test/Result_JPEG2000/TOTALE_MIX8.pkl','rb' ))
	res_jpg2000mix16 = pkl.load( open( '/home/ascalella/dataset/Test/Result_JPEG2000/TOTALE_MIX16.pkl','rb' ))
	psnr_pifs = [30.20514, 30.62405]
	ratio_pifs = [157.53, 146.28]
	similiar_pifs = [0.83203, 0.84178]

	
	for fi in file:
		res = pkl.load( open( path + '/' + fi,'rb' ))
		
		for i in range(11):
			psnr[:, i] += res[i].psnr[:,-1]
			wpsnr[:, i] += res[i].wpsnr[:,-1]
			ratio[:, i] += res[i].ratio[:,-1]
			similiar[:, i] += res[i].ssmi[:,-1]
		#for i in range(11):
	psnr_jpg = res_jpg.psnr
	wpsnr_jpg = res_jpg.wpsnr
	ratio_jpg = res_jpg.ratio
	similiar_jpg = res_jpg.ssmi
	psnr_jpg2000 = res_jpg2000.psnr
	wpsnr_jpg2000 = res_jpg2000.wpsnr
	ratio_jpg2000 = res_jpg2000.ratio
	similiar_jpg2000 = res_jpg2000.ssmi
	psnr_jpg2000mix8 = res_jpg2000mix8.psnr
	ratio_jpg2000mix8 = res_jpg2000mix8.ratio
	similiar_jpg2000mix8 = res_jpg2000mix8.ssmi
	psnr_jpg2000mix16 = res_jpg2000mix16.psnr
	ratio_jpg2000mix16 = res_jpg2000mix16.ratio
	similiar_jpg2000mix16 = res_jpg2000mix16.ssmi

	plt.xlabel('Compression Ratio', fontsize=16, fontweight='bold')
	plt.ylabel('PSNR', fontsize=16, fontweight='bold')
	plt.xticks(np.arange(0 , 171, step=10))
	plt.yticks(np.arange(25, 71, step=5))
	plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [PSNR: Original, DecodedNet + ResidualComp]' , fontweight='bold')
	plt.grid(True)
	plt.plot(ratio[0, :]/len(file), psnr[0,:]/len(file), 'gs-', label='Block Size: 8')
	plt.plot(ratio[1, :]/len(file), psnr[1,:]/len(file), 'bo-', label='Block Size: 16')
	plt.plot(ratio[2, :]/len(file), psnr[2,:]/len(file), 'rd-', label='Block Size: 32')
	plt.plot(ratio[3, :]/len(file), psnr[3,:]/len(file), 'k*-', label='Block Size: 64')
	plt.plot(np.mean(ratio_jpg, axis=0), np.mean(psnr_jpg, axis=0), 'c+-', label='JPEG')
	plt.plot(np.mean(ratio_jpg2000, axis=0), np.mean(psnr_jpg2000, axis=0), 'mx-', label='JPEG2000')
	plt.plot(ratio_pifs, psnr_pifs, label='PIFS+UNET')
	#plt.plot(np.mean(ratio_jpg2000mix16, axis=0), np.mean(psnr_jpg2000mix16, axis=0), label='MIXJPEG2000_16')



	plt.legend()
	figure = plt.gcf()
	figure.set_size_inches(12, 6)
	plt.savefig(out_dir + '/QualityPlotPSNR_'+d[-1]+'.png')
	plt.close()

	plt.xlabel('Compression Ratio', fontsize=16, fontweight='bold')
	plt.ylabel('wPSNR', fontsize=16, fontweight='bold')
	plt.xticks(np.arange(0 , 171, step=10))
	plt.yticks(np.arange(25, 71, step=5))
	plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [wPSNR: Original, DecodedNet + ResidualComp]' , fontweight='bold')
	plt.grid(True)
	plt.plot(ratio[0, :]/len(file), wpsnr[0,:]/len(file), 'gs-', label='Block Size: 8')
	plt.plot(ratio[1, :]/len(file), wpsnr[1,:]/len(file), 'bo-', label='Block Size: 16')
	plt.plot(ratio[2, :]/len(file), wpsnr[2,:]/len(file), 'rd-', label='Block Size: 32')
	plt.plot(ratio[3, :]/len(file), wpsnr[3,:]/len(file), 'k*-', label='Block Size: 64')
	plt.plot(np.mean(ratio_jpg, axis=0), np.mean(wpsnr_jpg, axis=0), 'c+-', label='JPEG')
	plt.plot(np.mean(ratio_jpg2000, axis=0), np.mean(wpsnr_jpg2000, axis=0), 'mx-', label='JPEG2000')
	plt.plot(ratio_pifs, [28.30078, 28.58618], label='PIFS+UNET')
	#plt.plot(np.mean(ratio_jpg2000mix16, axis=0), np.mean(psnr_jpg2000mix16, axis=0), label='MIXJPEG2000_16')



	plt.legend()
	figure = plt.gcf()
	figure.set_size_inches(12, 6)
	plt.savefig(out_dir + '/QualityPlotwPSNR_'+d[-1]+'.png')
	plt.close()

	plt.xlabel('Compression Ratio', fontsize=16, fontweight='bold')
	plt.ylabel('SSIM', fontsize=16, fontweight='bold')
	plt.xticks(np.arange(0 , 171, step=10))
	#plt.yticks(np.arange(25, 71, step=5))
	plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [SSIM: Original, DecodedNet + ResidualComp]' , fontweight='bold')
	plt.grid(True)
	plt.plot(ratio[0, :]/len(file), similiar[0,:]/len(file), 'gs-', label='Block Size: 8')
	plt.plot(ratio[1, :]/len(file), similiar[1,:]/len(file), 'bo-', label='Block Size: 16')
	plt.plot(ratio[2, :]/len(file), similiar[2,:]/len(file), 'rd-', label='Block Size: 32')
	plt.plot(ratio[3, :]/len(file), similiar[3,:]/len(file), 'k*-', label='Block Size: 64')
	plt.plot(np.mean(ratio_jpg, axis=0), np.mean(similiar_jpg, axis=0), 'c+-', label='JPEG')
	plt.plot(np.mean(ratio_jpg2000, axis=0), np.mean(similiar_jpg2000, axis=0), 'mx-', label='JPEG2000')
	plt.plot(ratio_pifs, similiar_pifs, label='PIFS+UNET')
	#plt.plot(np.mean(ratio_jpg2000mix8, axis=0), np.mean(similiar_jpg2000mix8, axis=0), label='MIXJPEG2000_8')
	#plt.plot(np.mean(ratio_jpg2000mix16, axis=0), np.mean(similiar_jpg2000mix16, axis=0), label='MIXJPEG2000_16')


	plt.legend()
	figure = plt.gcf()
	figure.set_size_inches(12, 6)
	plt.savefig(out_dir + '/QualityPlotSSIM_'+d[-1]+'.png')
	plt.close()


def plot_res3(path = None, out_dir = None):
	use('Agg')
	if path is None: path = '/home/ascalella/dataset/Test/Result'
	if out_dir is None: out_dir = '/home/ascalella/dataset/Test/Images'

	file = listdir(path)
	step = 0.5
	
	psnr = np.zeros((4,8))
	ratio = np.zeros((4,8))
	psnr_jpg = np.zeros((10))
	ratio_jpg = np.zeros((10))
	psnr_jpg2000 = np.zeros((10))
	ratio_jpg2000 = np.zeros((10))

	for fi in file:
		res = pkl.load( open( path + '/' + fi,'rb' ))
		res_jpg = pkl.load( open( path + '_JPEG/' + fi,'rb' ))
		res_jpg2000 = pkl.load( open( path + '_JPEG2000/' + fi,'rb' ))

		for i in range(8):
			psnr[:, i] += res[-1].psnr[:,i]
			ratio[:, i] += res[-1].ratio[:,i]
		#for i in range(11):
		psnr_jpg += res_jpg.psnr
		ratio_jpg += res_jpg.ratio
		psnr_jpg2000 += res_jpg2000.psnr
		ratio_jpg2000 += res_jpg2000.ratio
	plt.xlabel('Compression Ratio', fontsize=16, fontweight='bold')
	plt.ylabel('PSNR', fontsize=16, fontweight='bold')
	#plt.xticks(np.arange(0 , int((ratio_jpg2000/len(file)).max()), step=5))
	#plt.yticks(np.arange(int(((psnr)/len(file)).min()) - 5 , int(((psnr)/len(file)).max()) + 5, step=step))
	plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [PSNR: Original, DecodedNet + ResidualComp]' , fontweight='bold')
	plt.grid(True)
	plt.plot(ratio[0, :]/len(file), psnr[0,:]/len(file), 'gs-', label='Block Size: 8')
	plt.plot(ratio[1, :]/len(file), psnr[1,:]/len(file), 'bo-', label='Block Size: 16')
	plt.plot(ratio[2, :]/len(file), psnr[2,:]/len(file), 'rd-', label='Block Size: 32')
	plt.plot(ratio[3, :]/len(file), psnr[3,:]/len(file), 'k*-', label='Block Size: 64')
	#plt.plot(ratio_jpg/len(file), psnr_jpg/len(file), 'c+-', label='JPEG')
	#plt.plot(ratio_jpg2000/len(file), psnr_jpg2000/len(file), 'mx-', label='JPEG2000')


	plt.legend()
	figure = plt.gcf()
	figure.set_size_inches(12, 6)
	plt.savefig(out_dir + '/QualityPlotVS2.png')
	plt.close()


def testJPEG(img_path, quality = [10, 15,20,25,33,42,51,60,68,76,82,90,100]):
	ps = np.empty((len(quality)))
	wps = np.empty((len(quality)))
	rm = np.empty((len(quality)))
	ssi = np.empty((len(quality)))
	ratio = np.empty((len(quality)))
	original = np.array(imread(img_path, plugin='tifffile'), dtype='uint16')
	dim_org = stat(img_path).st_size
	im = Image.fromarray((original/256).astype('uint8'))
	x = 0
	strinf_f = ''
	for i in quality:
		im.save('/home/ascalella/dataset/Test/prova.jpeg', 'jpeg', quality=i)
		decoded = np.array(imread('/home/ascalella/dataset/Test/prova.jpeg'), dtype='uint16')
		dim_jpg = stat('/home/ascalella/dataset/Test/prova.jpeg').st_size
		cr = dim_org/dim_jpg
		psnr, rmse = Compute_PSNR(original/256, decoded)
		ss = ssim(original/65535., (decoded*256)/65535., data_range=1.0, multichannel=True)
		wpsnr= wPSNR(original/256, decoded)
		ssi[x] = ss
		ps[x] = psnr
		wps[x] = wpsnr
		rm[x] = rmse
		ratio[x] = cr
		strinf_f = strinf_f + 'Quality: {0}\n'.format(i)
		strinf_f = strinf_f + 'PSNR(Original, Decoded) = {0:.4f} \n'.format(psnr)
		strinf_f = strinf_f + 'WPSNR(Original, Decoded) = {0:.4f} \n'.format(wpsnr)
		strinf_f = strinf_f + 'RMSE(Original, Decoded) = {0:.4f} \n'.format(rmse)
		strinf_f = strinf_f + 'SSMI(Original, Decoded) = {0:.4f} \n'.format(ss)
		strinf_f = strinf_f + 'Dimension(Comp) = {0:.4f} bytes - {1:.4f} Kbytes\n'.format(dim_jpg, dim_jpg/1024.0)
		strinf_f = strinf_f + 'CompressionRatio(Original, Decoded) = {0:.4f} \n\n'.format(cr)
		x += 1
	result = Summary_JPEG(quality, ps, rm, ratio, ssi, wps)
	d = img_path.rsplit('/')
	upp = img_path.replace(d[-2]+ '/' +d[-1], '')

	#with open(upp + 'Report_JPEG/' + d[-1].replace('.tif', '_report.txt'), 'wt') as file:
		#file.write(strinf_f)

	return result


def testJPEG2000(img_path):
	ps = []
	wps = []
	rm = []
	ssi = []
	ratio = []
	original = np.array(imread(img_path, plugin='tifffile'), dtype='uint16')
	dim_org = stat(img_path).st_size

	strinf_f = ''
	for i in range(3,169,12):
		
		decoded = GL.Jp2k('/home/ascalella/dataset/Test/prova.jp2', data=original, cratios=[i])
		dim_jpg = stat('/home/ascalella/dataset/Test/prova.jp2').st_size
		cr = dim_org/dim_jpg
		psnr, rmse = Compute_PSNR(original, decoded[:])
		ss = ssim(original/65535., decoded[:]/65535., data_range=1.0, multichannel=True)
		wpsnr = wPSNR(original, decoded[:])
		ps.append(psnr)
		wps.append(wpsnr)
		rm.append(rmse)
		ssi.append(ss)
		ratio.append(cr)
		strinf_f = strinf_f + 'PSNR(Original, Decoded) = {0:.4f} \n'.format(psnr)
		strinf_f = strinf_f + 'WPSNR(Original, Decoded) = {0:.4f} \n'.format(wpsnr)
		strinf_f = strinf_f + 'RMSE(Original, Decoded) = {0:.4f} \n'.format(rmse)
		strinf_f = strinf_f + 'SSMI(Original, Decoded) = {0:.4f} \n'.format(ss)
		strinf_f = strinf_f + 'Dimension(Comp) = {0:.4f} bytes - {1:.4f} Kbytes\n'.format(dim_jpg, dim_jpg/1024.0)
		strinf_f = strinf_f + 'CompressionRatio(Original, Decoded) = {0:.4f} \n\n'.format(cr)

	result = Summary_JPEG(np.arange(3,169,12), np.asarray(ps), np.asarray(rm), np.asarray(ratio), np.asarray(ssi), np.asarray(wps))
	d = img_path.rsplit('/')
	upp = img_path.replace(d[-2]+ '/' +d[-1], '')

	with open(upp + 'Report_JPEG2000/' + d[-1].replace('.tif', '_report.txt'), 'wt') as file:
		file.write(strinf_f)

	return result


def MyCurve(img, n_coeff, blksize):

	if len(img.shape) > 2:
		img = rgb2gray(img)

	half = int(n_coeff/2)
	if np.mod(n_coeff,2) != 0:
		half = int((n_coeff+1)/2)

	co_p = 4
	#dctcoef = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
	n_out = (img.shape[0]/blksize)**2
	out = np.zeros((int(n_out),int(2 + co_p + n_coeff/2)), dtype='int32')
	x = 0
	for i in range(0, img.shape[0], blksize):
		for j in range(0, img.shape[1], blksize):
			dctcoef = dct2(img[i: i+blksize, j: j+blksize])
			zcoef = zigzag(dctcoef)
			sign = np.sign(zcoef[:n_coeff+1])
			sign = (sign+1)/2
			
			intsign = int(str(sign.astype(int))[1:-1].replace('\n','').replace(' ', ''),2)
			out[x,0] = intsign
			out[x,1] = np.abs(zcoef[0].astype(int))
			point = np.abs(zcoef[1:n_coeff+1])

			x_values = np.arange(1, n_coeff+1, 1)
			#popt, _ = curve_fit(objective, x_values, point[:n_coeff])
			popt = np.polyfit(x_values, point, 3)
			out[x,2:2+co_p] = np.rint(popt*10)
			dif = (point - np.polyval(np.rint(popt*10)/10., x_values))
			#out[x,2+co_p:2+co_p+half] = dif[:half].clip(-1.,1.).astype(int)
			#out[x,2+co_p+half:] = dif[half:].clip(-16.,15.).astype(int)
			out[x,2+co_p:] = dif[half:].clip(-16.,15.).astype(int)

			#out[x,2+co_p:2+co_p+half] = dif[:half].astype(int)
			#out[x,2+co_p+half:] = dif[half:].astype(int)

			#out[x,2+co_p:] = ((point[half] - np.polyval(np.rint(popt*10)/10., x_values[:half]))).astype(int)

			x = x + 1
	return out


def MyIcurve(code, im_sz, n_coeff, blksize):

	half = int(n_coeff/2)
	if np.mod(n_coeff,2) != 0:
		half = int((n_coeff+1)/2)
	co_p = 4

	len_c = int((im_sz/blksize)**2)
	mycode = np.reshape(code, (int(len_c), int(2 + co_p + n_coeff/2)))

	x = 0
	y = 0
	out = np.zeros((im_sz, im_sz))
	for i in range(0, len_c):
		dctcoef = np.zeros(blksize**2)
		list_sign = list(format(mycode[i,0], '0'+str(int(n_coeff+1))+'b'))
		arr_sign = np.asarray(list_sign, dtype='int8')*2 - 1
		dctcoef[0] = mycode[i,1]
		x_line = np.arange(1, n_coeff+1, 1)
		#a,b,c,d = mycode[i,2:6]/10.
		#y_line = objective(x_line, a, b, c, d)
		popt = mycode[i,2:6]/10.
		coef = np.polyval(popt, x_line)
		coef[half:] = coef[half:] + mycode[i, 6:]
		dctcoef[1:n_coeff + 1] = coef
		#dctcoef[half+1:n_coeff+1] = np.polyval(popt, x_line[half: n_coeff+1])
		#curve = np.poly1d(mycode[i,2:6]/10.)
		#curve2 = np.poly1d(mycode[i,6:]/10.)

		#dctcoef[1: half] = curve(np.arange(1, half))
		#dctcoef[half: n_coeff+1] = curve2(np.arange(half ,n_coeff+1))

		dctcoef[0:n_coeff+1] = dctcoef[0:n_coeff+1] * arr_sign
		idctcoef = inverse_zigzag(dctcoef, blksize, blksize)
		out[x: x+blksize, y:y+blksize] = idct2(idctcoef)
		y = y+blksize
		if y == im_sz:
			y = 0
			x = x + blksize
	return out


def MyCompressionCurve(img, n_coeff, blksize, out_dir):

	img = img.astype('float32')

	if np.max(img) > 255.:
		img = img/255.

	im_sz = img.shape[0]

	R = MyCurve(img[:,:, 0], n_coeff, blksize)
	G = MyCurve(img[:,:, 1], n_coeff, blksize)
	B = MyCurve(img[:,:, 2], n_coeff, blksize)

	off = 0
	fmt = 'b1u8u6u16'
	out = bytearray(round((calcsize(fmt) + 3*(n_coeff + 1 + 6 + 4 + 6 + 9 + 9 + int(n_coeff)/2 * 5)*(im_sz/blksize)**2) / 8.)+1) #int(n_coeff/2) * 5
	pack_into(fmt, out, off, 0, blksize, n_coeff, im_sz, fill_padding=False)
	off += calcsize(fmt)
	'''
	fmt = ('u'+str(n_coeff+1))*R.shape[0]
	pack_into(fmt, out, off, R[:, 0].ravel(), fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, R[:, 1], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's4'*R.shape[0]
	pack_into(fmt, out, off, R[:, 2], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, R[:, 3], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, R[:, 4], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, R[:, 5], fill_padding=False)
	off += calcsize(fmt)


	fmt = ('u'+str(n_coeff+1))*G.shape[0]
	pack_into(fmt, out, off, G[:, 0], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, G[:, 1], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's4'*R.shape[0]
	pack_into(fmt, out, off, G[:, 2], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, G[:, 3], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, G[:, 4], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, G[:, 5], fill_padding=False)
	off += calcsize(fmt)


	fmt = ('u'+str(n_coeff+1))*B.shape[0]
	pack_into(fmt, out, off, B[:, 0], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, B[:, 1], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's4'*R.shape[0]
	pack_into(fmt, out, off, B[:, 2], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's6'*R.shape[0]
	pack_into(fmt, out, off, B[:, 3], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, B[:, 4], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's9'*R.shape[0]
	pack_into(fmt, out, off, B[:, 5], fill_padding=False)
	off += calcsize(fmt)
	fmt = 's5'*R.shape[0]

	for j in range(int(n_coeff/2)):
		pack_into(fmt, out, off, R[:, 6 + j], fill_padding=False)
		off += calcsize(fmt)
		pack_into(fmt, out, off, G[:, 6 + j], fill_padding=False)
		off += calcsize(fmt)
		pack_into(fmt, out, off, B[:, 6 + j], fill_padding=False)
		off += calcsize(fmt)



	'''
	fmt = 'u'+str(n_coeff+1)+'s6s4s6s9s9'

	#fmt2 = 's2'*n_coeff/2+'s6'*n_coeff/

	for i in range(R.shape[0]):
		pack_into(fmt, out, off, R[i, 0], R[i, 1], R[i, 2], R[i, 3], R[i, 4], R[i, 5], fill_padding=False)
		off += calcsize(fmt)
		#pack_into('s2'*int(n_coeff/2), out, off, *R[i, 6 :6 + int(n_coeff/2)], fill_padding=False)
		#off += calcsize('s2'*int(n_coeff/2))
		pack_into('s5'*int(n_coeff/2), out, off, *R[i, 6:], fill_padding=False)
		off += calcsize('s5'*int(n_coeff/2))
		pack_into(fmt, out, off, G[i, 0], G[i, 1], G[i, 2], G[i, 3], G[i, 4], G[i, 5], fill_padding=False)
		off += calcsize(fmt)
		#pack_into('s2'*int(n_coeff/2), out, off, *G[i, 6 :6 + int(n_coeff/2)], fill_padding=False)
		#off += calcsize('s2'*int(n_coeff/2))
		pack_into('s5'*int(n_coeff/2), out, off, *G[i, 6:], fill_padding=False)
		off += calcsize('s5'*int(n_coeff/2))
		pack_into(fmt, out, off, B[i, 0], B[i, 1], B[i, 2], B[i, 3], B[i, 4], B[i, 5], fill_padding=False)
		off += calcsize(fmt)
		#pack_into('s2'*int(n_coeff/2), out, off, *B[i, 6 :6 + int(n_coeff/2)], fill_padding=False)
		#off += calcsize('s2'*int(n_coeff/2))
		pack_into('s5'*int(n_coeff/2), out, off, *B[i, 6:], fill_padding=False)
		off += calcsize('s5'*int(n_coeff/2))

	with open(out_dir + '.bin', 'wb') as file:
		file.write(out)
	
	return out


def MyDecompressionCurve(path_bin):

	with open(path_bin, 'rb') as file:
		raw = file.read()
	
	off = 0
	lossless = unpack_from('b1', raw, off)
	off += 1

	fmt = 'u8u6u16'
	blksize, n_coeff, im_sz = unpack_from(fmt, raw, off)
	off += calcsize(fmt)
	n_row = int((im_sz**2)/(blksize**2))
	half = int(n_coeff/2)
	Rh = np.empty((n_row,6+int(n_coeff/2)), dtype='int32')
	Gh = np.empty((n_row,6+int(n_coeff/2)), dtype='int32')
	Bh = np.empty((n_row,6+int(n_coeff/2)), dtype='int32')

	fmt = 'u'+str(n_coeff+1)+'s6s4s6s9s9'
	for i in range(n_row):

		Rh[i,:6] = unpack_from(fmt, raw, off)
		off += calcsize(fmt)
		#Rh[i, 6: 6 + half] = unpack_from('s2'*half, raw, off)
		#off += calcsize('s2'*half)
		Rh[i, 6:] = unpack_from('s5'*half, raw, off)
		off += calcsize('s5'*half)

		Gh[i,:6] = unpack_from(fmt, raw, off)
		off += calcsize(fmt)
		#Gh[i, 6: 6 + half] = unpack_from('s2'*half, raw, off)
		#off += calcsize('s2'*half)
		Gh[i, 6:] = unpack_from('s5'*half, raw, off)
		off += calcsize('s5'*half)

		Bh[i,:6] = unpack_from(fmt, raw, off)
		off += calcsize(fmt)
		#Bh[i, 6: 6 + half] = unpack_from('s2'*half, raw, off)
		#off += calcsize('s2'*half)
		Bh[i, 6:] = unpack_from('s5'*half, raw, off)
		off += calcsize('s5'*half)


	R = MyIcurve(Rh, im_sz, n_coeff, blksize)
	G = MyIcurve(Gh, im_sz, n_coeff, blksize)
	B = MyIcurve(Bh, im_sz, n_coeff, blksize)

	out = np.zeros((im_sz,im_sz,3), dtype='float32')
	out[:, :, 0] = R
	out[:, :, 1] = G
	out[:, :, 2] = B

	return out


def MyCompressionMix(img, blksize, quality, tresh, out_dir):

	img = img.astype('float32')

	if np.max(img) > 255.:
		img = img/256.

	if quality > 10:
		quality = 10.
	elif quality <=0:
		quality = 1.

	im_sz = img.shape[0]
	
	
	#R_t, R = rec_mask(img[:, :, 0], im_sz, quality*10, tresh)
	#G_t, G = rec_mask(img[:, :, 1], im_sz, quality*10, tresh)
	#B_t, B = rec_mask(img[:, :, 2], im_sz, quality*10, tresh)

	R, R_t = MyDct2(img[:, :, 0], blksize, quality*10, tresh)
	G, G_t = MyDct2(img[:, :, 1], blksize, quality*10, tresh)
	B, B_t = MyDct2(img[:, :, 2], blksize, quality*10, tresh)


	#len_r = len(R_t)

	#len_g = len(G_t)

	#len_b = len(B_t)
	
	Rin = np.asarray(rle.encode(R), dtype='int32')
	Gin = np.asarray(rle.encode(G), dtype='int32')
	Bin = np.asarray(rle.encode(B), dtype='int32')
	#T = np.concatenate((R_t, G_t, B_t))
	tree = np.asarray(rle.encode(np.concatenate((R_t, G_t, B_t))), dtype='int32')

	RGB = np.concatenate((Rin.ravel(), Gin.ravel(), Bin.ravel(), tree.ravel()))

	dic = huffman_code(Counter(RGB.ravel()))
	codeR = bt.bitarray()
	codeR.encode(dic, Rin.ravel())
	codeG = bt.bitarray()
	codeG.encode(dic, Gin.ravel())
	codeB = bt.bitarray()
	codeB.encode(dic, Bin.ravel())
	codeT = bt.bitarray()
	codeT.encode(dic, tree.ravel())


	totLenVal = 0
	for i in dic.keys(): totLenVal += len(dic[i])

	key = list(dic.keys())
	value = list(dic.values())

	a = int(np.max(key))
	b = int(np.min(key))
	c = int(np.max([a, np.abs(b)]))
	n_bk = c.bit_length() + 1


	off = 0
	fmt = 'u8u4u16u32u32u32u32u16u5'
	out = bytearray(round((calcsize(fmt) + len(codeR) + len(codeG) + len(codeB)+ len(codeT) + len(dic)*(n_bk + 5) + totLenVal) / 8.)+1)
	pack_into(fmt, out, off, blksize, quality, im_sz, len(codeR), len(codeG), len(codeB), len(codeT), len(dic), n_bk, fill_padding=False)
	off += calcsize(fmt)
	#print(n_bk)

	for i in range(len(dic)):
		fmt = 'u5s'+str(n_bk)+'r'+str(len(value[i]))
		pack_into(fmt, out, off, len(value[i]), key[i], value[i], fill_padding=False)
		off += calcsize(fmt)

	pack_into('r'+str(len(codeR))+'r'+str(len(codeG))+'r'+str(len(codeB))+'r'+str(len(codeT)), out, off, codeR, codeG, codeB, codeT,fill_padding=False)

	with open(out_dir + '.bin', 'wb') as file:
			file.write(out)

	return out

def MyDecompressionMix(path_bin):

	with open(path_bin, 'rb') as file:
		raw = file.read()
	

	blksize, quality, im_sz, lenR, lenG, lenB, lenT, len_dic, n_bk  = unpack_from('u8u4u16u32u32u32u32u16u5', raw, 0)
	off = calcsize('u8u4u16u32u32u32u32u16u5')
	dic ={}
	for i in range(len_dic):
		len_k, val = unpack_from('u5s'+str(n_bk), raw, off)
		off += calcsize('u5s'+str(n_bk))
		key = unpack_from('r'+str(len_k), raw, off)
		off += calcsize('r'+str(len_k))
		k = bt.bitarray()
		k.frombytes(bytes(key[0]))
		dic[val] = k[:len_k]


	Rr, Gr, Br, Tr = unpack_from('r'+str(lenR)+'r'+str(lenG)+'r'+str(lenB)+'r'+str(lenT),raw, off)

	tree = bt.decodetree(dic)
	Rb = bt.bitarray()
	Gb = bt.bitarray()
	Bb = bt.bitarray()
	Tb = bt.bitarray()

	Rb.frombytes(bytes(Rr))
	Gb.frombytes(bytes(Gr))
	Bb.frombytes(bytes(Br))
	Tb.frombytes(bytes(Tr))

	Rh = Rb[:lenR].decode(tree)
	Gh = Gb[:lenG].decode(tree)
	Bh = Bb[:lenB].decode(tree)
	Th = Tb[:lenT].decode(tree)

	Rh = np.reshape(Rh, (2, int(len(Rh)/2)))
	Gh = np.reshape(Gh, (2, int(len(Gh)/2)))
	Bh = np.reshape(Bh, (2, int(len(Bh)/2)))
	Th = np.reshape(Th, (2, int(len(Th)/2)))

	Rr = rle.decode(Rh[0,:], Rh[1,:])
	Gr = rle.decode(Gh[0,:], Gh[1,:])
	Br = rle.decode(Bh[0,:], Bh[1,:])
	Tr = rle.decode(Th[0,:], Th[1,:])

	n_bloc = int((im_sz/blksize)**2)
	#R = Irec_mask(Tr[:lr], Rr, im_sz, quality)[0]
	#G = Irec_mask(Tr[lr: lr+lg], Gr, im_sz, quality)[0]
	#B = Irec_mask(Tr[lr+lg:], Br, im_sz, quality)[0]
	R = MyIdct2(Rr, blksize, quality, Tr[:n_bloc])
	G = MyIdct2(Gr, blksize, quality, Tr[n_bloc: 2*n_bloc])
	B = MyIdct2(Br, blksize, quality, Tr[2*n_bloc: 3*n_bloc])

	out = np.zeros((im_sz,im_sz,3), dtype='float32')
	out[:, :, 0] = R
	out[:, :, 1] = G
	out[:, :, 2] = B

	return out


def MyDct2(img, blksize, quality, tresh):

	if len(img.shape) > 2:
		img = rgb2gray(img)

	quantization_value = np.array([[16,  11,  10,  16,  24,   40,   51,   61],
									 [12,  12,  14,  19,  26,   58,   60,   55],
									 [14,  13,  16,  24,  40,   57,   69,   56],
									 [14,  17,  22,  29,  51,   87,   80,   62],
									 [18,  22,  37,  56,  68,   109,  103,  77],
									 [24,  35,  55,  64,  81,   104,  113,  92],
									 [49,  64,  78,  87,  103,  121,  120,  101],
									 [72,  92,  95,  98,  112,  100,  103,  99]], dtype='float32')


	if quality < 50:
		s = 5000./quality
	elif quality >= 50 and quality < 100:
		s = 200. - 2.*quality
	else:
		s = 1.

	q_mtx = (s*quantization_value + 50.)/100.


	if blksize > 8:
		blkq_mtx = np.ones((blksize, blksize), dtype='float32')
		blkq_mtx[:8, :8] = q_mtx
	else:
		blkq_mtx = q_mtx

	dctcoef = []
	res = []
	for i in range(0, img.shape[0], blksize):
		for j in range(0, img.shape[1], blksize):
			if np.linalg.norm(img[i: i+blksize, j: j+blksize]) > tresh:
				dctcoef += list(zigzag(np.rint(dct2(img[i: i+blksize, j: j+blksize]) / blkq_mtx)))
				res.append(1)
			else:
				res.append(0)



	return np.asarray(dctcoef, dtype='int32'), np.asarray(res, dtype='int32')

def MyIdct2(cuttcoef, blksize, quality, res):

	quantization_value = np.array([[16,  11,  10,  16,  24,   40,   51,   61],
									 [12,  12,  14,  19,  26,   58,   60,   55],
									 [14,  13,  16,  24,  40,   57,   69,   56],
									 [14,  17,  22,  29,  51,   87,   80,   62],
									 [18,  22,  37,  56,  68,   109,  103,  77],
									 [24,  35,  55,  64,  81,   104,  113,  92],
									 [49,  64,  78,  87,  103,  121,  120,  101],
									 [72,  92,  95,  98,  112,  100,  103,  99]], dtype='float32')


	if quality < 50.:
		s = 5000./quality
	elif quality >= 50. and quality < 100.:
		s = 200. - 2.*quality
	else:
		s = 1.

	q_mtx = (s*quantization_value + 50.)/100.

	if blksize > 8:
		blkq_mtx = np.ones((blksize, blksize), dtype='float32')
		blkq_mtx[:8, :8] = q_mtx
	else:
		blkq_mtx = q_mtx
	x = 0
	y = 0
	img = np.zeros((int(len(res)/blksize), int(len(res)/blksize)), dtype='float32')
	for i in range(0, img.shape[0],blksize):
		for j in range(0, img.shape[1],blksize):
			#print(x, i, j)
			if res[x] == 1:
				img[i: i+blksize, j: j+blksize] = idct2(inverse_zigzag(cuttcoef[y: y + 64], blksize, blksize)* blkq_mtx)
				x+=1
				y+=64
			else:
				x+=1

	return img

def wPSNR(a, b):
	a = a.astype('float32')
	b = b.astype('float32')
	#a_b = np.expand_dims(a, axis=0)
	#b_b = np.expand_dims(b, axis=0)
	#gax, gay = image_gradients(convert_to_tensor(a_b))
	#gbx, gby = image_gradients(convert_to_tensor(b_b))
	gax= np.gradient(a, axis=0)
	gay = np.gradient(a, axis=1)
	gbx = np.gradient(b, axis=0)
	gby = np.gradient(b, axis=1)

	
	wa = 0.5*(np.abs(gax)+np.abs(gbx))
	wa = 1.+(1+wa)/(1+wa.ravel().max())
	wb = 0.5*(np.abs(gay)+np.abs(gby))
	wb = 1.+(1+wb)/(1+wb.ravel().max())

	df = wa*wb*(a-b)
	mse = np.sum(df.ravel()**2)/len(df.ravel())
	wpsnr = 10*np.log10((a.max()**2)/mse)

	return wpsnr
