import os
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime

from .data import *
from .network import *


def get_neg_sample(pos, num_item, exclude=None):
    pos = list(set(pos))
    pos_len = len(pos)
    if exclude:
        pos = exclude + pos
    pos = sorted(pos)
    
    sample = np.arange(0, num_item - len(pos))
    pos_adj = pos - np.arange(len(pos))
    search = np.searchsorted(pos_adj, sample, side='right')
    neg = sample + search
    return neg


def build_network(config, dataloader, train):
    network = Network(config)
    x, y = dataloader.__getitem__(0)
    _ = network(x, training=False)

    if train == 'train':
        optimizer = tf.keras.optimizers.Adam()
        print(network.summary())
        return network, optimizer
    else:
        ckpt_dir = os.path.join(config['base_dir'], 'model', config['model_name'], 'ckpt', f'{config["data_name"]}')
        ckpt = [i for i in os.listdir(ckpt_dir)]
        val_loss = [float(i.split('val_')[1].split('.h5')[0]) for i in ckpt]
        weights = ckpt[np.argmin(val_loss)]
        weights = os.path.join(ckpt_dir, weights)
        network.load_weights(weights)
        return network


@tf.function
def train_step(network, optimizer, x, y):
    with tf.GradientTape() as tape:
        output_mask = x[1][1]
        pred = network(x, training=True)
        implicit_loss = tf.keras.losses.sparse_categorical_crossentropy(y[0], pred[0])
        implicit_loss *= output_mask
        explicit_loss = tf.keras.losses.mse(y[1], pred[1])
        explicit_loss *= output_mask
        loss = 0.7*implicit_loss + 0.3*explicit_loss

    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    loss = tf.reduce_mean(loss)
    return loss

def val_step(network, x, y):
    output_mask = x[1][1]
    pred = network(x, training=True)
    implicit_loss = tf.keras.losses.sparse_categorical_crossentropy(y[0], pred[0])
    implicit_loss *= output_mask
    explicit_loss = tf.keras.losses.mse(y[1], pred[1])
    explicit_loss *= output_mask
    loss = implicit_loss + explicit_loss
    loss = tf.reduce_mean(loss)
    return loss

def test_step(network, x, y, popularity, test_method, config):
    logit = network(x, training=False)[0]
    logit = np.array(logit[:,-1,:])
    y = y[0]
    pos = y[:,-1,0]

    if test_method == 'complete':
        rank = np.argsort(np.argsort(-logit))
        rank = [i[j] for i,j in zip(rank, pos)]
    else:
        neg = [get_neg_sample(i, config['num_item'], exclude=[0,1]) for i in y[:,:,0]]
        if test_method == 'sampling':
            neg = np.array([np.random.choice(i, size=100, replace=False) for i in neg])
        elif test_method == 'weighted_sampling':
            neg_prob = [popularity[i].values for i in neg]
            neg_prob = [i / i.sum() for i in neg_prob]
            neg = np.array([np.random.choice(i, size=100, p=j, replace=False) for i,j in zip(neg, neg_prob)])

        sample = np.append(pos[:,None], neg, axis=-1)
        total_logit = np.array([logit[idx][i] for idx, i in enumerate(sample)])
        rank = np.argsort(np.argsort(-total_logit))[:,0]
    
    rank = [i + 1 for i in rank]
    return rank

def print_train_log(step, start_time, train_loss):
    time_delta = datetime.today() - start_time
    time = f'{time_delta.seconds // 60}M {time_delta.seconds % 60}S'
    print(f'step : {str(step).zfill(5)} | train loss : {train_loss:.3f} | time : {time}')

def train(config, resume):
    model_dir = os.path.join(config['base_dir'], 'model', config['model_name'])
    data_dir =  os.path.join(model_dir, 'data', config['data_name'])
    ckpt_dir = os.path.join(config['base_dir'], 'model', config['model_name'], 'ckpt', config['data_name'])
        
    train_data = load_pickle(os.path.join(data_dir, 'train.pkl'))
    train_loader = DataLoader(train_data, 'train', config)
    val_data = load_pickle(os.path.join(data_dir, 'val.pkl'))
    val_loader = DataLoader(val_data, 'val', config)
    
    network, optimizer = build_network(config, train_loader, 'train')
    if resume:
        ckpt_dir = os.path.join(config['base_dir'], 'model', config['model_name'], 'ckpt', f'{config["data_name"]}')
        ckpt_path = os.listdir(ckpt_dir)[-1]
        start_st = int(ckpt_path[3:8])
        network.load_weights( os.path.join(ckpt_dir, ckpt_path))
    else:
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)
        start_st = 0

    start_time = datetime.today()
    for train_st in range(start_st+1, config['train_step_size']+1):
        train_x, train_y = train_loader.next()
        train_loss = train_step(network, optimizer, train_x, train_y)         
        
        if train_st % 100 == 0:
            print_train_log(train_st, start_time, train_loss)

            val_loss = 0
            for val_st in range(config['val_step_size']):
                val_x, val_y = val_loader.next()
                val_loss += val_step(network, val_x, val_y)
            val_loss /= config['val_step_size']
            network.save_weights(os.path.join(ckpt_dir, f'st_{str(train_st).zfill(5)}_train_{train_loss:.3f}_val_{val_loss:.3f}.h5'))  

def test(config, test_method, num_test):
    model_dir = os.path.join(config['base_dir'], 'model', config['model_name'])
    data_dir =  os.path.join(model_dir, 'data', config['data_name'])
    test_data = load_pickle(os.path.join(data_dir, 'test.pkl'))
    test_loader = DataLoader(test_data, 'test', config)
    popularity = load_pickle(os.path.join(data_dir, 'popularity.pkl'))
    
    network = build_network(config, test_loader, 'test')
    test_step_size = round(num_test / config['batch_size'])

    total_rank = []
    for _ in range(test_step_size):
        test_x, test_y = test_loader.next()
        rank = test_step(network, test_x, test_y, popularity, test_method, config)
        total_rank += rank

    total_rank = total_rank[:num_test]
    hr1 = np.mean([i == 1 for i in total_rank])
    hr5 = np.mean([i <= 5 for i in total_rank])
    hr10 = np.mean([i <= 10 for i in total_rank])
    hr50 = np.mean([i <= 50 for i in total_rank])

    print('-' * 70)
    print(f'HR@1 : {hr1:.3f} | HR@5 : {hr5:.3f} | HR@10 : {hr10:.3f} | HR@50 : {hr50:.3f}')
    print('-' * 70)