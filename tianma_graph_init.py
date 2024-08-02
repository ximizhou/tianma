import argparse
import os
import numpy as np
from utils import save_dataset, load_dataset, save_kg
from Graph_generate.lastfm_data_process import LastFmDataset
from Graph_generate.lastfm_star_data_process import LastFmStarDataset
from Graph_generate.lastfm_graph import LastFmGraph
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='LAST_FM', choices=['LAST_FM', 'LAST_FM_STAR', 'YELP', 'YELP_STAR'],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    args = parser.parse_args()
    
    # Dataset and Graph dictionary mappings
    DatasetDict = {
        'LAST_FM': LastFmDataset,
        'LAST_FM_STAR': LastFmStarDataset,
        'YELP': YelpDataset,
        'YELP_STAR': YelpDataset,
    }
    GraphDict = {
        'LAST_FM': LastFmGraph,
        'LAST_FM_STAR': LastFmGraph,
        'YELP': YelpGraph,
        'YELP_STAR': YelpGraph,
    }

    # Load dataset
    print(f'Loading {args.data_name} from file...')
    if not os.path.isdir(TMP_DIR[args.data_name]):
        os.makedirs(TMP_DIR[args.data_name])
    dataset = DatasetDict[args.data_name](DATA_DIR[args.data_name])
    save_dataset(args.data_name, dataset)
    print(f'Saved {args.data_name} dataset successfully!')

    # Generate graph
    print(f'Creating {args.data_name} graph...')
    dataset = load_dataset(args.data_name)
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    print(f'Saved {args.data_name} graph successfully!')

def construct_graph(kg):
    users = kg.G['user'].keys()
    items = kg.G['item'].keys()
    features = kg.G['feature'].keys()
    num_nodes = len(users) + len(items) + len(features)
    graph = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < len(users) and j < len(users) + len(items):
                graph[i][j] = 1
                graph[j][i] = 1
            elif len(users) <= i < len(users) + len(items):
                if j - len(users) - len(items) in kg.G['item'][i - len(users)]['belong_to']:
                    graph[i][j] = 1
                    graph[j][i] = 1

    print(graph)
    return graph

if __name__ == '__main__':
    main()
