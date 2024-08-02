import pickle
import numpy as np
import random
import torch
import os
import sys

# Dataset names
LAST_FM = 'LAST_FM'
LAST_FM_STAR = 'LAST_FM_STAR'
YELP = 'YELP'
YELP_STAR = 'YELP_STAR'

# Directories for datasets and temporary files
DATA_DIRECTORIES = {
    LAST_FM: './data/lastfm',
    YELP: './data/yelp',
    LAST_FM_STAR: './data/lastfm_star',
    YELP_STAR: './data/yelp',
}
TEMP_DIRECTORIES = {
    LAST_FM: './tmp/last_fm',
    YELP: './tmp/yelp',
    LAST_FM_STAR: './tmp/last_fm_star',
    YELP_STAR: './tmp/yelp_star',
}

def cuda_(variable):
    return variable.cuda() if torch.cuda.is_available() else variable

def save_dataset(dataset_name, dataset_object):
    dataset_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'dataset.pkl')
    with open(dataset_file, 'wb') as file:
        pickle.dump(dataset_object, file)

def load_dataset(dataset_name):
    dataset_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'dataset.pkl')
    with open(dataset_file, 'rb') as file:
        dataset_object = pickle.load(file)
    return dataset_object

def save_knowledge_graph(dataset_name, knowledge_graph):
    kg_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'kg.pkl')
    with open(kg_file, 'wb') as file:
        pickle.dump(knowledge_graph, file)

def load_knowledge_graph(dataset_name):
    kg_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'kg.pkl')
    with open(kg_file, 'rb') as file:
        knowledge_graph = pickle.load(file)
    return knowledge_graph

def save_graph(dataset_name, graph):
    graph_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'graph.pkl')
    with open(graph_file, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(dataset_name):
    graph_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'graph.pkl')
    with open(graph_file, 'rb') as file:
        graph = pickle.load(file)
    return graph

def load_embeddings(dataset_name, embedding_name, epoch):
    if embedding_name:
        path = os.path.join(TEMP_DIRECTORIES[dataset_name], 'embeds', f'{embedding_name}.pkl')
    else:
        return None
    with open(path, 'rb') as file:
        embeddings = pickle.load(file)
        print(f'{embedding_name} Embedding loaded successfully!')
        return embeddings

def load_rl_agent(dataset_name, filename, epoch_user):
    model_file = os.path.join(TEMP_DIRECTORIES[dataset_name], 'RL-agent', f'{filename}-epoch-{epoch_user}.pkl')
    model_dict = torch.load(model_file)
    print(f'RL policy model loaded from {model_file}')
    return model_dict

def save_rl_agent(dataset_name, model, filename, epoch_user):
    model_directory = os.path.join(TEMP_DIRECTORIES[dataset_name], 'RL-agent')
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
    model_file = os.path.join(model_directory, f'{filename}-epoch-{epoch_user}.pkl')
    torch.save(model, model_file)
    print(f'RL policy model saved at {model_file}')

def save_rl_metrics(dataset_name, filename, epoch, success_rate, time_spent, mode='train'):
    log_directory = os.path.join(TEMP_DIRECTORIES[dataset_name], 'RL-log-merge')
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    log_file = os.path.join(log_directory, f'{filename}.txt')
    with open(log_file, 'a') as file:
        if mode == 'train':
            file.write('=========== Train ===============\n')
            file.write(f'Starting {epoch} user epochs\n')
            file.write(f'training SR@5: {success_rate[0]}\n')
            file.write(f'training SR@10: {success_rate[1]}\n')
            file.write(f'training SR@15: {success_rate[2]}\n')
            file.write(f'training Avg@T: {success_rate[3]}\n')
            file.write(f'training hDCG: {success_rate[4]}\n')
            file.write(f'Spending time: {time_spent}\n')
            file.write('================================\n')
        elif mode == 'test':
            file.write('=========== Test ===============\n')
            file.write(f'Testing {epoch} user tuples\n')
            file.write(f'Testing SR@5: {success_rate[0]}\n')
            file.write(f'Testing SR@10: {success_rate[1]}\n')
            file.write(f'Testing SR@15: {success_rate[2]}\n')
            file.write(f'Testing Avg@T: {success_rate[3]}\n')
            file.write(f'Testing hDCG: {success_rate[4]}\n')
            file.write(f'Testing time: {time_spent}\n')
            file.write('================================\n')

def save_rl_model_log(dataset_name, filename, epoch, epoch_loss, train_length):
    log_directory = os.path.join(TEMP_DIRECTORIES[dataset_name], 'RL-log-merge')
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    log_file = os.path.join(log_directory, f'{filename}.txt')
    with open(log_file, 'a') as file:
        file.write(f'Starting {epoch} epoch\n')
        file.write(f'training loss : {epoch_loss / train_length}\n')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    device_ids = [int(device_id) for device_id in args.gpu.split()]
    device = torch.device(f"cuda:{device_ids[0]}") if use_cuda else torch.device("cpu")
    return device, device_ids
