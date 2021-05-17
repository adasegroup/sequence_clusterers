def get_raw_data(path):
    with open(path+'/train.pkl', 'rb') as f:
        train_data = pickle.load(f, encoding = 'latin1')
    with open(path+'/test.pkl', 'rb') as f:
        test_data = pickle.load(f, encoding = 'latin1')
    with open(path+'/dev.pkl', 'rb') as f:
        dev_data = pickle.load(f, encoding = 'latin1')
        
    meta_data = train_data['args']
    dim_process = train_data['dim_process']
    train_data_no_meta = train_data['train']
    test_data_no_meta = test_data['test']
    dev_data_no_meta = dev_data['dev']
        
    return {'meta_data': meta_data,
            'dim_process': dim_process,
            'train': train_data_no_meta,
            'test': test_data_no_meta,
            'dev': dev_data_no_meta}

def preprocess_data(raw_data):
    tmp = {'train': [],
           'test': [],
           'dev': []}
    dim_size = raw_data['dim_process']
    #train
    tmp['train'] = [[(0.0, torch.eye(dim_size+1)[:,0])]+\
                     [(j['time_since_start'], torch.eye(dim_size+1)[:,j['type_event']+1]) for j in i] for i in raw_data['train']]
    #test
    tmp['test'] = [[(0.0, torch.eye(dim_size+1)[:,0])]+\
                    [(j['time_since_start'], torch.eye(dim_size+1)[:,j['type_event']+1]) for j in i] for i in raw_data['test']]
    #dev
    tmp['dev'] = [[(0.0, torch.eye(dim_size+1)[:,0])]+\
                   [(j['time_since_start'], torch.eye(dim_size+1)[:,j['type_event']+1]) for j in i] for i in raw_data['dev']]
    return tmp