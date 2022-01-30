#import UNET
#import Pix2Pix
#import prova as Pix2Pix
#from tensorflow.keras.models import load_model
#import PIFS
#import numpy as np
#import time
#import CompressionResidual as CR
import Library as LB
from os import environ, listdir
#import pickle as pkl
#from multiprocessing import Pool
#from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#from tensorflow import make_ndarray
#import sys
environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
environ["CUDA_VISIBLE_DEVICES"]="-1" # per usare la prima GPU che ha ID 0

#epoch = ['050', '070', '100', '120', '150']


#for i in epoch:
#	LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint/unet_0'+ i +'.h5')

#Pix2Pix.train_pix(100, 0) 
#UNET.train_unet(150, 0, 0)
#environ["CUDA_VISIBLE_DEVICES"]="-1" # per usare la prima GPU che ha ID 0
LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint_LAST/unet_0030.h5')
#LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint_riccioPix/generator100.h5')
#LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint_riccioPix/generator120.h5')
#LB.predict(net_path='/home/ascalella/dataset/Network/Checkpoint/unet_0070.h5')

'''
dir_org = '/home/ascalella/dataset/Test/Tiles_original/'
dir_img = '/home/ascalella/dataset/Test/T1'
file = listdir(dir_img)
for i in range(0, len(file), 10):
  proc = []
  pool = Pool(processes=9)
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+1]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+2]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+3]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+4]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+5]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+6]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+7]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+8]), )))
  proc.append(pool.apply_async(CR.comprTest, (dir_org + str(file[i+9]), )))
  CR.comprTest(dir_org + str(file[i]))
  for j in proc:
    a = j.get()
  pool.close()
  pool.join()
  print(i)
'''