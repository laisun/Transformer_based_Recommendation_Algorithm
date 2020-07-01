from src import data, train



CONFIG = {
    'model_name' : 'Context_Feedback',
    'train' : 'train',
    'resume' : False,
    'data_name' : 'ML1M',
    'base_dir' : '',
    'train_step_size' : 10000,
    'val_step_size' : 100,
    'batch_size' : 256,
    'mask_rate' : 0.2, 
    'drop_rate' : 0.2, 
    'num_user' : 6040,
    'num_item' : 3706, 
    'num_head' : 2, 
    'num_layer' : 2, 
    'seq_len' : 100,
    'model_dim' : 256,
    'dff' : 256,
}

def main():
    if CONFIG['train'] == 'train':
        data.prepare(CONFIG)
        train.train(CONFIG, False)
    else:
        train.test(CONFIG, 'weighted_sampling', 100)

if __name__ == '__main__':
    main()