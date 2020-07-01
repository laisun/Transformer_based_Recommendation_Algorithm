import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf

def check_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def save_pickle(data, pkl_path):
    check_dir(os.path.dirname(pkl_path))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def load_raw_data(base_dir, data_name):
    if data_name == 'ML20M':
        rating = pd.read_csv(os.path.join(base_dir, f'data/{data_name}/rating.csv'))
        rating.columns = ['userId', 'itemId', 'rating', 'timestamp']
        rating['timestamp'] = pd.to_datetime(rating['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    elif data_name == 'ML10M':
        rating = pd.read_table(os.path.join(base_dir, f'data/{data_name}/rating.dat'), sep='::', engine='python', header=None)
        rating.columns = ['userId', 'itemId', 'rating', 'timestamp']
        rating['timestamp'] = rating['timestamp'].apply(datetime.fromtimestamp)
    
    elif data_name == 'ML1M':
        rating = pd.read_table(os.path.join(base_dir, f'data/{data_name}/rating.dat'), sep='::', engine='python', header=None)
        rating.columns = ['userId', 'itemId', 'rating', 'timestamp']
        rating['timestamp'] = rating['timestamp'].apply(datetime.fromtimestamp)
        
    rating = rating.sort_values(['userId', 'timestamp'])
    return rating

def _prepare(rating):
    rating['userId'] = rating['userId'].apply(lambda x : f'User_{x}')
    rating['itemId'] = rating['itemId'].apply(lambda x : f'Item_{x}')
        
    user2id = {j:i for i,j in enumerate(rating['userId'].unique())}
    item2id = {j:i+2 for i,j in enumerate(rating['itemId'].unique())} # 0:PAD 1:MASK
    
    rating['userId'] = rating['userId'].apply(lambda x : user2id[x])
    rating['itemId'] = rating['itemId'].apply(lambda x : item2id[x])
    popularity = rating.groupby('itemId')['rating'].count()

    seq = rating.groupby('userId')[['itemId', 'rating', 'timestamp']].agg(lambda x : list(x))
    train = seq.apply(lambda x : x.agg(lambda y : y[:-2]), axis=1)
    val = seq.apply(lambda x : x.agg(lambda y : y[:-1]), axis=1)
    test = seq
    return seq, train, val, test, user2id, item2id, popularity

def prepare(config):
    print('prepare started')
    rating = load_raw_data(config['base_dir'], config['data_name'])
    prep_data = _prepare(rating)
    prep_data_name = ['seq', 'train', 'val', 'test', 'user2id', 'item2id', 'popularity']
    save_dir = os.path.join(config['base_dir'], 'model', config['model_name'], 'data', config['data_name'])
    for d, n in zip(prep_data, prep_data_name):
        save_pickle(d, os.path.join(save_dir, f'{n}.pkl'))
    print('prepare completed')


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, train, config):
        self.data = data
        self.train = train
        self.config = config
        self.len_data = len(self.data)
        self.idx = 0
        self.on_epoch_end()

    def __len__(self):
        return np.ceil(self.len_data / self.config['batch_size']).astype(np.int32)
    
    def on_epoch_end(self):
        self.indices = np.random.permutation(self.len_data)
        # self.indices = np.arange(self.len_data)

    def prepare_batch(self, batch):
        x_item = np.array([pad_seq(i[:-1], self.config['seq_len'], 'left') for i in batch['itemId']])
        x_year = [[j.year - 1990 for j in i] for i in batch['timestamp']]
        x_year = np.array([pad_seq(i[:-1], self.config['seq_len'], 'left') for i in x_year])
        x_month = [[j.month for j in i] for i in batch['timestamp']]
        x_month = np.array([pad_seq(i[:-1], self.config['seq_len'], 'left') for i in x_month])
        x_day = [[j.day for j in i] for i in batch['timestamp']]
        x_day = np.array([pad_seq(i[:-1], self.config['seq_len'], 'left') for i in x_day])
        x_hour = [[j.hour + 1 for j in i] for i in batch['timestamp']]
        x_hour = np.array([pad_seq(i[:-1], self.config['seq_len'], 'left') for i in x_hour])

        implicit = [pad_seq(i[1:], self.config['seq_len'], 'left') for i in batch['itemId']]
        implicit = np.array(implicit)[:,:,None]
        explicit = [pad_seq(i[1:], self.config['seq_len'], 'left') for i in batch['rating']]
        explicit = np.array(explicit)[:,:,None]
        explicit = explicit / 5
        
        padding_mask = np.array([[j==0 for j in i] for i in x_item])[:, None, None, :]
        look_ahead_mask = create_look_ahead_mask(self.config['seq_len'])
        input_mask = padding_mask + look_ahead_mask
        input_mask = tf.cast(tf.math.not_equal(input_mask, 0), tf.float32)
        output_mask = np.array([[j>0 for j in i] for i in x_item], dtype=np.float32)

        x = (x_item, x_year, x_month, x_day, x_hour)
        y = (implicit, explicit)
        mask = (input_mask, output_mask)
        return (x, mask), y

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.config['batch_size'] : (idx+1)*self.config['batch_size']]
        batch = self.data.iloc[batch_idx]
        (x, mask), y = self.prepare_batch(batch)
        return (x, mask), y
    
    def next(self):
        if self.idx == self.__len__():
            self.on_epoch_end()
            self.idx = 0
        (x, mask), y = self.__getitem__(self.idx)
        self.idx += 1
        return (x, mask), y

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = mask[None, None, :, :]
    return mask  # (1, 1, seq_len, seq_len)

def pad_seq(seq, seq_len, where):
    seq = seq[-seq_len:]
    if where == 'left':
        seq = np.pad(seq, (seq_len-len(seq), 0), 'constant')
    elif where == 'right':
        seq = np.pad(seq, (0, seq_len-len(seq)), 'constant')
    return seq
