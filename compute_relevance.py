import logging
import numpy as np
import os
import yaml
import tqdm
import multiprocessing
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from model.text_similarity_utils import mpnet_similarity, rouge
import pandas as pd
import torch

import model.launch.prepare  # NOQA

import pandas as pd

def mine_textual_queries(dataset):
    data = [(dataset[i]['keyid'], dataset[i]['text']) for i in range(len(dataset))]
    df = pd.DataFrame(data, columns=['keyid', 'text'])

    # drop duplicated queries
    # df = df.drop_duplicates(subset=['text'])

    # get queries and its ids
    candidate_queries, candidate_queries_idx = df['text'].tolist(), df.index.tolist()

    return candidate_queries, candidate_queries_idx

def get_motions_and_associated_descriptions(dataset):
    data = [(dataset[i]['keyid'], dataset[i]['text']) for i in range(len(dataset))]
    df = pd.DataFrame(data, columns=['keyid', 'text'])

    # drop duplicated (motion, desc) and 
    # df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)

    # group by motion
    df = df.groupby(['keyid'], sort=False, as_index=False).agg({'text':lambda x: list(x), 'index': 'min'})

    motions, agg_desc, motions_ids = df['keyid'].tolist(), df['text'].tolist(), df['index'].tolist()

    return motions, agg_desc, motions_ids

def compute_relevances_wrt_query(query):
    i, query_caption = query

    if any(compute_relevances_wrt_query.npy_file[i, :] < 0):

        # init the scorer
        if compute_relevances_wrt_query.method == 'rougeL':
            scorer = rouge.Rouge()
        else:
            scorer = mpnet_similarity.MPNetSimilarity(compute_relevances_wrt_query.sent_model)

        # get motions and associated descriptions
        motions, agg_desc, motions_ids = get_motions_and_associated_descriptions(compute_relevances_wrt_query.dataset)

        # find the indexes of the exact-matching motions
        indexes_of_exact_result = [any([query_caption in a for a in b]) for b in agg_desc]
        indexes_of_exact_result = np.asarray(indexes_of_exact_result).nonzero()[0]

        # run the scorer and retrieve the relevances
        relevances = scorer.compute_score(agg_desc, query_caption)

        # patch the spice relevances with the exact results
        relevances[indexes_of_exact_result] = 1.0

        # save on npy file
        compute_relevances_wrt_query.npy_file[i, :] = relevances
        

def compute_relevances_wrt_query_single_thread(query, method, agg_desc_embs, npy_file, sent_model=None):
    i, query_caption = query

    if any(npy_file[i, :] < 0):

        # init the scorer
        if method == 'rougeL':
            scorer = rouge.Rouge()
        else:
            scorer = mpnet_similarity.MPNetSimilarity(sent_model)

        # get motions and associated descriptions
        agg_desc, _ = agg_desc_embs

        # find the indexes of the exact-matching motions
        indexes_of_exact_result = [any([query_caption in a for a in b]) for b in agg_desc]
        indexes_of_exact_result = np.asarray(indexes_of_exact_result).nonzero()[0]

        # run the scorer and retrieve the relevances
        relevances = scorer.compute_score(agg_desc_embs, query_caption)

        # patch the spice relevances with the exact results
        relevances[indexes_of_exact_result] = 1.0

        # save on npy file
        npy_file[i, :] = relevances

def compute_relevances_wrt_motion_single_thread(motion_queries, method, candidate_queries_embds, npy_file, sent_model=None):
    i, query_captions = motion_queries

    if any(npy_file[i, :] < 0):
        # init the scorer
        if method == 'rougeL':
            scorer = rouge.Rouge()
        else:
            scorer = mpnet_similarity.MPNetSimilarity(sent_model)

        # get motions and associated descriptions
        candidate_queries, _ = candidate_queries_embds
        relevances = np.zeros((len(query_captions), len(npy_file[i])))
        for j, query_caption in enumerate(query_captions):
            # find the indexes of the exact-matching motions
            indexes_of_exact_result = [query_caption == a for a in candidate_queries]
            indexes_of_exact_result = np.asarray(indexes_of_exact_result).nonzero()[0]

            # run the scorer and retrieve the relevances
            relevances[j] = scorer.compute_score(candidate_queries_embds, query_caption)

            # patch the spice relevances with the exact results
            relevances[j][indexes_of_exact_result] = 1.0

        # save on npy file
        npy_file[i, :] = relevances.max(axis=0)

def parallel_worker_init(npy_file, dataset, method, sent_model):
    compute_relevances_wrt_query.npy_file = npy_file
    compute_relevances_wrt_query.dataset = dataset
    compute_relevances_wrt_query.method = method
    compute_relevances_wrt_query.sent_model = sent_model

@hydra.main(version_base=None, config_path="configs", config_name="compute_relevance")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    data_cfg = cfg.data
    type = cfg.type
    dset_name = data_cfg.dataname 

    dataset = instantiate(data_cfg, split=cfg.split)

    sent_model = instantiate(cfg.sent_model)

    # get queries
    candidate_queries, candidate_queries_idx = mine_textual_queries(dataset)

    # get motions and associated descriptions
    motions, agg_desc, motions_ids = get_motions_and_associated_descriptions(dataset)

    relevance_dir = 'outputs/computed_relevances'
    if not os.path.exists(relevance_dir):
        os.makedirs(relevance_dir)
    relevance_filename = os.path.join(relevance_dir, '{}-{}-{}-{}.npy'.format(dset_name, cfg.split, cfg.method, type))
    if os.path.isfile(relevance_filename):
        answ = input("Relevances for {} already existing in {}. Continue? (y/n)".format(cfg.method, relevance_filename))
        if answ != 'y':
            quit()

    # filename = os.path.join(cache_dir,'d_{}.npy'.format(query_img_index))
    n_queries = len(candidate_queries)
    n_motions = len(motions)
    if type == "t2m":
        shape = (n_queries, n_motions)
    elif type == "m2t":
        shape = (n_motions, n_queries)
    else:
        return NotImplementedError()

    if os.path.isfile(relevance_filename):
        # print('Graph distances file existing for image {}, cache {}! Loading...'.format(query_img_index, cache_name))
        print('Loading existing file {} with shape {} x {}'.format(relevance_filename, n_queries, n_motions))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=shape, mode='r+')
    else:
        print('Creating new file {} with shape {} x {}'.format(relevance_filename, n_queries, n_motions))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=shape, mode='w+')
        npy_file[:, :] = -1 

    # pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=n).start()
    print('Starting relevance computation...')
    # with multiprocessing.Pool(processes=cfg.ncpus, initializer=parallel_worker_init,
    #                           initargs=(npy_file, dataset, cfg.method, sent_model)) as pool:
    #     for _ in tqdm.tqdm(pool.imap_unordered(compute_relevances_wrt_query, enumerate(candidate_queries)), total=n_queries):
    #         pass
    if type == "t2m":
        agg_desc_embs = []
        # preload all sentence embeddings
        for m_desc in tqdm.tqdm(agg_desc, total=n_motions):
            agg_desc_embs.append(sent_model(m_desc))
        for query in tqdm.tqdm(enumerate(candidate_queries), total=n_queries):
            compute_relevances_wrt_query_single_thread(query, cfg.method, (agg_desc, agg_desc_embs), npy_file, sent_model=sent_model)
    else:
        query_embs = []
        # preload all sentence embeddings
        for query in tqdm.tqdm(candidate_queries, total=n_queries):
            query_embs.append(sent_model([query]))
        for motion_queies in tqdm.tqdm(enumerate(agg_desc), total=n_motions):
            compute_relevances_wrt_motion_single_thread(motion_queies, cfg.method, (candidate_queries, query_embs), npy_file, sent_model=sent_model)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
