'''
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) (https://doi.org/10.1145/3331184.3331267)

Run this file in terminal with arguments, per example:
>> run.py --dataset Gowella --emb_dim 64 --layers [64]

authors: Mohammed Yusuf Noor, Muhammed Imran Özyar, Calin Vasile Simon
'''

import pandas as pd
import torch

import os
from time import time
from datetime import datetime

from utils.load_data import Data
from utils.parser import parse_args
from utils.helper_functions import early_stopping,\
                                   train,\
                                   split_matrix,\
                                   compute_ndcg_k,\
                                   eval_model
from ngcf import NGCF

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

if __name__ == '__main__':

    # read parsed arguments
    args = parse_args()
    data_dir = args.data_dir
    dataset = args.data_size
    batch_size = args.batch_size
    layers = eval(args.layers)
    emb_dim = args.emb_dim
    lr = args.lr
    reg = args.reg
    mess_dropout = args.mess_dropout
    node_dropout = args.node_dropout
    k = args.k

    # generate the NGCF-adjacency matrix
    data_generator = Data(path=data_dir + dataset, batch_size=batch_size)
    adj_mtx = data_generator.get_adj_mat()

    # create model name and save
    modelname =  "NGCF" + \
        "_bs_" + str(batch_size) + \
        "_nemb_" + str(emb_dim) + \
        "_layers_" + str(layers) + \
        "_nodedr_" + str(node_dropout) + \
        "_messdr_" + str(mess_dropout) + \
        "_reg_" + str(reg) + \
        "_lr_"  + str(lr)

    # create NGCF model
    model = NGCF(data_generator.n_users, 
                 data_generator.n_items,
                 emb_dim,
                 layers,
                 reg,
                 node_dropout,
                 mess_dropout,
                 adj_mtx)
    if use_cuda:
        model = model.cuda()

    # current best metric
    cur_best_metric = 0

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set values for early stopping
    cur_best_loss, stopping_step, should_stop = 1e3, 0, False
    today = datetime.now()

    print("Start at " + str(today))
    print("Using " + str(device) + " for computations")
    print("Params on CUDA: " + str(next(model.parameters()).is_cuda))

    results = {"Epoch": [],
               "Loss": [],
               "Recall": [],
               "NDCG": [],
               "Training Time": []}
    

    for epoch in range(args.n_epochs):

        t1 = time()
        loss = train(model, data_generator, optimizer)
        training_time = time()-t1
        print("Epoch: {}/{}, Training time: {:.2f}s, Loss: {:.4f}".
            format(epoch, args.n_epochs, training_time, loss))

        # print test evaluation metrics every N epochs (provided by args.eval_N)
        
        if should_stop == True: 
            print('Training is ended successfully.')
            break
    print('Training is ended successfully.')

    # save
    
