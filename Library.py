import Pix2Pix
import UNET
import PIFS
import CompressionResidual as CR
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from matplotlib import use
from os import listdir, stat
from skimage.metrics import structural_similarity as ssim
import glymur as GL
from PIL import Image


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

def predict(net_path = None, dir_img = None, dir_test = None, out_pre=None, n_img=0):

	if dir_test is None: dir_test = '/home/ascalella/dataset/Test/Tiles_decoded_FILTER24'
	if dir_img is None: dir_img = '/home/ascalella/dataset/Test/Tiles_original'
	if net_path is None: net_path = '/home/ascalella/dataset/Network/Checkpoint/unet_0070.h5'#'/home/ascalella/dataset/Network/PSNR10/unet_10_70.h5'
	if out_pre is None: out_pre = '/home/ascalella/dataset/Test/Tiles_net_LAST'

	name_list = []
	test_list = []
	image_list = []
	files_blurred = listdir(dir_test)
	if n_img == 0:
		n_img = len(files_blurred)
	for filepath,k in zip(files_blurred,range(n_img)):
		flag = PIFS.imread(dir_test + '/{0}'.format(filepath), plugin='tifffile')
		#flag = PIFS.np.zeros((512,512,3))
		#flag[:,:,0] = my_imresize(flag1[:,:,0])
		#flag[:,:,1] = my_imresize(flag1[:,:,1])
		#flag[:,:,2] = my_imresize(flag1[:,:,2])     
		test_list.append(flag)
		#test_list.append(PIFS.imread(dir_test + '/{0}'.format(filepath), plugin='tifffile'))
		image_list.append(PIFS.imread(dir_img + '/{0}'.format(filepath), plugin='tifffile'))
		name_list.append(filepath)

	test = PIFS.np.asarray(test_list, dtype='float32')
	image = PIFS.np.asarray(image_list, dtype='float32')

	net = UNET.load_model(net_path, custom_objects={'MyLoss': UNET.MyLoss})
	res_net = net.predict(test / 65535.)
	#res_net = net.predict((test/ 32767.5) - 1.)
	ck = 0.
	res_ps = []
	res_wps = []
	res_rm = []
	res_ss = []
	i = 0
	mean_B = 0
	mean_N = 0
	mean_wB = 0
	mean_wN = 0
	mean_rB = 0
	mean_rN = 0
	mean_ssB = 0
	mean_ssN = 0
	for im, org, blu in zip(res_net, image, test):
	  flag = NormalizeData(im) * 65535.0
	  #flag = PIFS.np.zeros((512,512,3))
	  #flag[:,:,0] = my_imresize(flag1[:,:,0])
	  #flag[:,:,1] = my_imresize(flag1[:,:,1])
	  #flag[:,:,2] = my_imresize(flag1[:,:,2])
	  #flag = im.clip(0., 65535.)
	  PIFS.imsave(out_pre + '/' + name_list[i], standardize(org, flag).astype('uint16'), plugin='tifffile');
	  ck += CR.Compute_PSNR(org, flag)[0]
	  pB, rB = CR.Compute_PSNR(org, standardize(org, blu))
	  pN, rN = CR.Compute_PSNR(org, standardize(org, flag))
	  ssB = ssim(org/65535., standardize(org, blu)/65535., data_range=1.0, multichannel=True)
	  ssN = ssim(org/65535., standardize(org, flag)/65535., data_range=1.0, multichannel=True)
	  wpB = CR.wPSNR(org, standardize(org, blu))
	  wpN = CR.wPSNR(org, standardize(org, flag))
	  res_ps.append([pN, pB, pN - pB])
	  res_wps.append([wpN, wpB, wpN - wpB])
	  res_rm.append([rN, rB, rN - rB])
	  res_ss.append([ssN, ssB, ssN - ssB])
	  mean_B += pB
	  mean_N += pN
	  mean_wB += wpB
	  mean_wN += wpN
	  mean_rB += rB
	  mean_rN += rN
	  mean_ssB += ssB
	  mean_ssN += ssN
	  i = i + 1

	a = np.asarray(res_ps)
	b = np.asarray(res_rm)
	c = a[:,0]
	d = b[:,0]
	e = np.asarray(res_ss)
	f = e[:,0]
	g = np.asarray(res_wps)
	h = g[:,0]

	res_ps.append([mean_N/i, mean_B/i, (mean_N - mean_B)/i])
	res_ps.append([c.mean(), c.std(), c.var()])
	res_wps.append([mean_wN/i, mean_wB/i, (mean_wN - mean_wB)/i])
	res_wps.append([h.mean(), h.std(), h.var()])
	res_rm.append([mean_rN/i, mean_rB/i, (mean_rN - mean_rB)/i])
	res_rm.append([d.mean(), d.std(), d.var()])
	res_ss.append([mean_ssN/i, mean_ssB/i, (mean_ssN - mean_ssB)/i])
	res_ss.append([f.mean(), f.std(), f.var()])
	print(ck/i)

	#res = PIFS.np.asarray(res_list, dtype='float32')
	#res_pspy = UNET.tf.image.psnr(image/ 65535., res/ 65535., max_val=1.0)
	d = net_path.rsplit('/')
	PIFS.np.savetxt('/home/ascalella/dataset/Result/Prova/PSNR_'+d[-2]+'_'+d[-1].replace('h5', 'txt'), res_ps, delimiter=', ', fmt='%3.5f')
	PIFS.np.savetxt('/home/ascalella/dataset/Result/Prova/WPSNR_'+d[-2]+'_'+d[-1].replace('h5', 'txt'), res_wps, delimiter=', ', fmt='%3.5f')
	PIFS.np.savetxt('/home/ascalella/dataset/Result/Prova/RMSE_'+d[-2]+'_'+d[-1].replace('h5', 'txt'), res_rm, delimiter=', ', fmt='%3.5f')
	PIFS.np.savetxt('/home/ascalella/dataset/Result/Prova/SSMI_'+d[-2]+'_'+d[-1].replace('h5', 'txt'), res_ss, delimiter=', ', fmt='%3.5f')



def plot_res(path = None):
	use('Agg')
	if path is None: path = '/home/ascalella/dataset/Test/Result'
	file = listdir(path)
	for i in range(10):
	  psnr = np.zeros((4,8))
	  ratio = np.zeros((4,8))
	  for fi in file:
	    res = pkl.load( open( path + '/' + fi,'rb' ))
	    psnr += res[i].psnr
	    ratio += res[i].ratio
	  plt.xlabel('Compression Ratio')
	  plt.ylabel('PSNR')
	  plt.title('[CR: Original / (PIFScode + ResidualComp)] --- [PSNR: Original, DecodedNet + ResidualComp]')
	  #plt.axis([0, 110, 20, 50])
	  #plt.text(10, 22, 'Quality: ' + str(i+1))
	  plt.grid(True)
	  plt.plot(ratio/len(file), psnr/len(file))
	  figure = plt.gcf()
	  figure.set_size_inches(10, 6)
	  plt.savefig('/home/andrea/PycharmProjects/HOPE/Plot/QualityPlot'+ str(i+1) + '.png', dpi=500)
	  plt.close()

def standardize(a,b):
	a= a.astype('float32')
	b= b.astype('float32')

	b[:,:,0] = (((b[:,:,0] - b[:,:,0].mean())/b[:,:,0].std()) * a[:,:,0].std()) + a[:,:,0].mean()
	b[:,:,1] = (((b[:,:,1] - b[:,:,1].mean())/b[:,:,1].std()) * a[:,:,1].std()) + a[:,:,1].mean()
	b[:,:,2] = (((b[:,:,2] - b[:,:,2].mean())/b[:,:,2].std()) * a[:,:,2].std()) + a[:,:,2].mean()

	return b.clip(0, a.max())

def multiple_plot(images, rows = 1, cols=1, out_dir=None, sup_title=None, sub_title=None):
	figure, ax = plt.subplots(nrows=rows,ncols=cols, constrained_layout=True)
	figure.set_size_inches(12, 6)
	for ind,title in enumerate(images):
		ax.ravel()[ind].imshow(images[title])
		ax.ravel()[ind].set_title(title, fontsize=16)
		if sub_title is not None:
			ax.ravel()[ind].get_xaxis().set_ticks([])
			ax.ravel()[ind].get_yaxis().set_ticks([])
			ax.ravel()[ind].set_xlabel(sub_title[ind], fontsize=16, fontweight='bold')
	if sup_title is not None:
		figure.suptitle(sup_title, fontsize=18, fontweight='bold')
	if out_dir is None:
		plt.show()
	else:
		plt.savefig(out_dir, dpi=200)
	plt.close()

def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())

def plot_result():
	dir_org = '/home/ascalella/dataset/Test/Tiles_original/'
	dir_net = '/home/ascalella/dataset/Test/Tiles_net_PSNR100SSIM/'
	dir_blur = '/home/ascalella/dataset/Test/Tiles_decoded_NEWNEW8/'
	dir_img = '/home/ascalella/dataset/Test/T2'
	dir_plot = '/home/ascalella/dataset/Test/Images/'
	quality = [7]
	n_coeff = [64]
	blksize = [16]
	for file in listdir(dir_img):
		img = np.asarray(PIFS.imread(dir_org + file, plugin='tifffile'), dtype='float32')
		img_net = np.asarray(PIFS.imread(dir_net + file, plugin='tifffile'), dtype='float32')
		blur = np.asarray(PIFS.imread(dir_blur + file, plugin='tifffile'), dtype='float32')
		dif = img - img_net
		for qt in quality:
			for blk in blksize:
				for n_co in n_coeff:
					fname = file.replace('.tif', '_q{0}_b{1}_c{2}'.format(qt, blk, n_co))
					out_dir = '/home/ascalella/dataset/Test/Pifs_bin/' + fname
					#a = CR.MyCompression(dif, n_co, blk, qt, 0, out_dir)
					#es = CR.MyDecompression(out_dir + '.bin')
					#res = standardize(img ,(img_net + es*256.))

					psnr = []
					jp2 = GL.Jp2k('/home/ascalella/dataset/Test/prova.jp2', data=img.astype('uint16'), cratios=[26])
					im = Image.fromarray((img/256).astype('uint8'))
					im.save(out_dir+'JPEG'+file.replace('.tif', '.jpeg'), 'jpeg', quality=9)
					dec = np.asarray(PIFS.imread(out_dir+'JPEG'+file.replace('.tif', '.jpeg')), dtype='float32')*256.


					psnr.append('Inf - 1.0')
					psnr.append('{0:.4f} - {1:.4f}'.format(CR.Compute_PSNR(img, blur)[0], ssim(img/65535., blur/65535., data_range=1.0, multichannel=True)))
					psnr.append('{0:.4f} - {1:.4f}'.format(CR.Compute_PSNR(img, img_net)[0], ssim(img/65535., img_net/65535., data_range=1.0, multichannel=True)))
					#psnr.append('{0:.4f} - {1:.4f}'.format(CR.Compute_PSNR(img, res)[0], ssim(img/65535., res/65535., data_range=1.0, multichannel=True)))
					psnr.append('{0:.4f} - {1:.4f}'.format(CR.Compute_PSNR(img, dec)[0], ssim(img/65535., dec/65535., data_range=1.0, multichannel=True)))
					#psnr.append('{0:.4f} - {1:.4f}'.format(CR.Compute_PSNR(img, jp2[:])[0], ssim(img/65535., jp2[:]/65535., data_range=1.0, multichannel=True)))
					dict_i = {}
					dict_i['Original'] = NormalizeData(img)
					dict_i['Pifs'] = NormalizeData(blur)
					dict_i['Unet'] = NormalizeData(img_net)
					dict_i['JPEG'] = NormalizeData(dec)
					dim_diff = stat(out_dir + '.bin').st_size
					dim_org = stat(dir_org + file).st_size
					dim_pifs = 9986
					sup_title = ' Quality: {0}    Block Size: {1}   N.Coefficients: {2} \n \n \n Compression Ratio: {3:.4f}'.format(qt, blk, n_co, dim_org/(dim_pifs+dim_diff))
					multiple_plot(dict_i, 1, 4, dir_plot + 'STAT'+ fname + '.png', sup_title, psnr)

