from os import environ, listdir
environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
environ["CUDA_VISIBLE_DEVICES"]="3" # per usare la prima GPU che ha ID 0
import UNET
#import Pix2Pix
#from tensorflow.keras.models import load_model
#import PIFS
#import numpy as np
#import time
#import CompressionResidual as CR
#mport Library as LB
#from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#from tensorflow import make_ndarray
#import sys



#event_acc = EventAccumulator('/home/ascalella/dataset/Train_Result/Log/fit/20200809-011747/')
#event_acc.Reload()
# Show all tags in the log file
#print(event_acc.Tags())

#xy_l2_loss = [(make_ndarray(s.tensor_proto)) for s in event_acc.Tensors('gen_gan_loss')]

#print(xy_l2_loss)

import sys
#sys.stderr = open('/home/ascalella/dataset/Train_Result/err.txt', 'w')
#start_n = time.time()
firstarg=sys.argv[1:]
#print(firstarg)
#PIFS.generate_tiles_new(firstarg, 0)

#PIFS.generate_tiles_final(['Prova'],0)
#start = time.time()
#PIFS.generate_tiles(['Prova2'],0)
#end = time.time()

#print('Normal: ', end - start, '\n Numba: ', start - start_n)


#R = PIFS.pkl.load( open( '/home/ascalella/dataset/Train/Original/Prova_res/b001_0_0_2_0_0_0.pkl', 'rb' ) )
#G = PIFS.pkl.load( open( '/home/ascalella/dataset/Train/Original/Prova_res/b001_0_0_2_0_0_1.pkl', 'rb' ) )
#B = PIFS.pkl.load( open( '/home/ascalella/dataset/Train/Original/Prova_res/b001_0_0_2_0_0_2.pkl', 'rb' ) )
#PIFS.pifs2bin([R,G,B], '/home/ascalella/dataset/Train/Original/Prova_res/b001_0_0_2_0_0_0')
#dir_org = '/home/ascalella/dataset/Test/Tiles_original/'
#dir_img = '/home/ascalella/dataset/Test/Benign/'
#for file, in PIFS.listdir(dir_net):
	#print(file)
	#CR.comprTest(dir_org + file)
	#PIFS.pkl.dump(pifs, open('/home/ascalella/dataset/Test/Result/' + file.replace('.tif','.pkl'), 'wb'), PIFS.pkl.HIGHEST_PROTOCOL)

#import Library as Lb
#Lb.plot_res()
	#img = np.asarray(PIFS.imread(dir + '/{0}'.format(file), plugin='tifffile'), dtype='uint16')
	#out_dir = '/home/ascalella/dataset/Test/Bin/' + file.replace('.tif','')
	#t = time.time()
	#a = CR.MyCompression(img, 64, 16, 10, 0, out_dir)
	#d = time.time()
	#es = CR.MyDecompression(out_dir+'.bin')
	#end = time.time()
	#dec_t = end - d
	#com_t = d - t
	#PIFS.imsave(dir + '/{0}'.format(file), es.astype('uint16') * 255., plugin='tifffile')
	#print('time com: ', com_t)
	#print('time dec: ', dec_t)


#LB.predict()
UNET.train_unet2(int(firstarg[0]), 0, 0)

#a = PIFS.imread('/home/ascalella/dataset/Test/b007.tif', plugin='tifffile')
#pifs = PIFS.pifs_encode_wsi(a.astype('float32'), 512, 16, 8)

#pifs = PIFS.pkl.load(open('/home/ascalella/dataset/Train_Result/pifs_wsi007.pkl', 'rb'))

#res = PIFS.pifs_decode_wsi(pifs, None, 1.0)

#PIFS.imsave('/home/ascalella/dataset/Train_Result/res007.tif', res, plugin='tifffile')

#res_net = PIFS.pifs_decode_wsi(pifs, Pix2Pix.get_pix(), 1.0)


#b = PIFS.imread('/home/ascalella/dataset/Train/Original/Tiles_decoded/b007_0_0_2.tif', plugin='tifffile')
#b = b.astype('float32')
#net = UNET.get_unet()
#b = b.reshape(64,64,64,3)
#res_net = net.predict(b/65535.)
#res_net = net.predict(PIFS.np.expand_dims((b / 65535.), axis=0))
#res_net = res_net.reshape(512,512,3)
#res_net = PIFS.NormalizeData(res_net) * 65535.0;
#PIFS.imsave('/home/ascalella/dataset/Train_Result/res007U_net.tif',  res_net.astype('uint16'), plugin='tifffile');

#Pix2Pix.train_pix(150, 0) 
#net = Pix2Pix.get_pix()
#res_net = net.predict(PIFS.np.expand_dims((b / 32767.5) - 1., axis=0))
#res_net = ((res_net) + 1.) * 32767.5
#PIFS.imsave('/home/ascalella/dataset/Train_Result/res007Pix_net.tif',  res_net.astype('uint16'), plugin='tifffile')

#epochL = ['30', '50', '70', '100', '120']
#epochP = ['30', '50', '70', '100']
#epoch = ['030', '050', '070', '100', '120']

#LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint_8/unet_0100.h5')
#LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint/unet_0150.h5')

#for i in epoch:
	#LB.predict(net_path='/home/ascalella/dataset/Network/PSNR10/generator'+ i +'.h5')
	#LB.predict(net_path='/home/ascalella/dataset/Network/PSNR10/unet_0'+ i +'.h5')

#LB.predict(net_path='/home/ascalella/dataset/Network/PSNR10/unet_0070.h5')
'''
sub = ['MAE', 'MSE', 'RMSE', 'PSNR_1', 'PSNR10']

for di in sub:
	for i in epochP:
		LB.predict(net_path='/home/ascalella/dataset/Network/'+ di +'/generator'+ i +'.h5')
		#LB.predict(net_path='/home/ascalella/dataset/Network/'+ di +'/unet_0'+ i +'.h5')
	print(di)
'''