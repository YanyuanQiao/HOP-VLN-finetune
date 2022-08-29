import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,BTokenizer,padding_idx,timeSince,preprocess_get_pano_states,current_best
from env import R2RBatch
from pytorch_transformers import BertForMaskedLM,BertTokenizer
from agent import Seq2SeqAgent
from feature import Feature
from eval import Evaluation
import json
import copy
import pdb
from ipdb import set_trace
from tensorboardX import SummaryWriter
from param import args
import torch.distributed as dist
# For philly
philly = False

metrics = ['length','nav_error','oracle success_rate','success_rate','spl','oracle path_success_rate','goal_progress']

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
feedback_method = args.feedback  # teacher or sample

print(args); print('')

def reduce_tensor(tensor,dst=0):
    rt = tensor.clone()
    dist.reduce(rt, dst, op=dist.ReduceOp.SUM)
    return rt

def create_folders(path):
    """ recursively create folders """
    if not os.path.isdir(path):
        while True:
            try:
                os.makedirs(path)
            except:
                pass
                time.sleep(1)
            else:
                break


TRAIN_VOCAB = 'tasks/NDH/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/NDH/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/NDH/results/'
SNAPSHOT_DIR = 'tasks/NDH/snapshots/'
PLOT_DIR = 'tasks/NDH/plots/'

if philly:
    RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'tasks/NDH/results/')
    PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'),'tasks/NDH/plots/')
    SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'tasks/NDH/snapshots/')
    TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, 'train_vocab.txt')
    TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, 'trainval_vocab.txt')
    print("using philly, output are rest")

    print('RESULT_DIR', RESULT_DIR)
    print('PLOT_DIR', PLOT_DIR)
    print('SNAPSHOT_DIR', SNAPSHOT_DIR)
    print('TRAIN_VOC', TRAIN_VOCAB)

#IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
IMAGENET_FEATURES = 'img_features/ResNet-152-places365.tsv'
FEATURE_SIZE = 2048
FEATURE_ALL_SIZE = 2176 # 2048

# Training settings.
agent_type = 'seq2seq'

# Fixed params from MP.
features = IMAGENET_FEATURES
batch_size = 8
word_embedding_size = 256
action_embedding_size = 32
target_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005


#def train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
#    log_every=100, val_envs=None, args=None):
def train(train_env, tok, n_iters, path_type, history, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
    log_every=100, val_envs=None):
   

    writer = SummaryWriter(log_dir=log_dir)
    #set_trace()
    agent = Seq2SeqAgent(train_env, "", tok, max_episode_len)

    record_file = open('./logs/' + args.name + '.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    ''' Train on training set, validating on both seen and unseen. '''
    if val_envs is None:
        val_envs = {}

    start_iter = 0
    if args.load is not None:
        if args.aug is not None:
            start_iter = agent.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = agent.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

#    best_val = {'val_seen': {"dist_to_end_reduction": 0., "state":"", 'update':False},
#            'val_unseen': {"dist_to_end_reduction": 0., "state":"", 'update':False}}
    best_val = {'val_unseen': {"dist_to_end_reduction": 0., "state":"", 'update':False}}

    start = time.time()

    best_dr = 0
    best_spl = 0
    best_iter = 0
    best_dr_iter = 0
    best_sr = 0
    myidx = 0
    split_string = "-".join(train_env.splits)
    #set_trace()
    for idx in range(start_iter, start_iter + n_iters, log_every):
        data_log = defaultdict(list)

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        myidx += interval
        print("PROGRESS: {}%".format(round((myidx) * 100 / n_iters, 4)))

        # Train for log_every interval
        agent.env = train_env
        agent.train(interval, feedback=feedback_method)


        # Log the training stats to tensorboard
        total = max(sum(agent.logs['total']), 1)
        length = max(len(agent.logs['critic_loss']), 1)
        critic_loss = sum(agent.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(agent.logs['entropy']) / total #/ length / args.batchSize
        RL_loss = sum(agent.logs['RL_loss']) / max(len(agent.logs['RL_loss']), 1)
        IL_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)
        entropy = sum(agent.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total, ", max_length", length)

        # Run validation
        loss_str = "iter {}".format(iter)
        #set_trace()
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env

            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=False, feedback='argmax', iters=None)

            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.write_results()
            #set_trace()

            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                #set_trace()
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['dist_to_end_reduction']:                    
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['dist_to_end_reduction']:
                            best_val[env_name]['dist_to_end_reduction'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['dist_to_end_reduction']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['dist_to_end_reduction'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)

        record_file = open('./logs/' + args.name + '.txt', 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                agent.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            else:
                agent.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        print("EVALERR: {}%".format(best_dr))

        if iter % 50 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    agent.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

def setup(action_space=-1, navigable_locs_path=None):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(RESULT_DIR):
        create_folders(RESULT_DIR)
    if not os.path.exists(PLOT_DIR):
        create_folders(PLOT_DIR)
    if not os.path.exists(SNAPSHOT_DIR):
        create_folders(SNAPSHOT_DIR)
    if not os.path.exists(navigable_locs_path):
        create_folders(navigable_locs_path)

    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)

    if navigable_locs_path:
        #if philly:
        #    navigable_locs_path = os.path.join(os.getenv('PT_OUTPUT_DIR'), "tasks/NDH/data")
        #    if not os.path.exists(navigable_locs_path):
        #        create_folders(navigable_locs_path)

        navigable_locs_path += '/navigable_locs.json'

        print('navigable_locs_path', navigable_locs_path)
    preprocess_get_pano_states(navigable_locs_path)
    global nav_graphs
    nav_graphs = None
    if action_space == -1:  # load navigable location cache
        with open(navigable_locs_path, 'r') as f:
            nav_graphs = json.load(f)
    return nav_graphs

#test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind, args)
def test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind, args):
    ''' Train on combined training and validation sets, and generate test submission. '''
    nav_graphs = setup(args.action_space, args.navigable_locs_path)
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    feature_store = Feature(features, args.panoramic)
    #set_trace()
    # Generate test submission
    test_env = R2RBatch(feature_store, nav_graphs, args.panoramic,args.action_space,batch_size=args.batch_size, splits=['test'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

#    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok,
#                        path_type=path_type, history=history, blind=blind)
    agent = Seq2SeqAgent(test_env, "", tok, max_episode_len)
    start_iter = agent.load(os.path.join(args.load))
    print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))


    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    #set_trace()
    agent.test_sub(use_dropout=False, feedback='argmax')
    agent.write_results()


def btest_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind,args):
    ''' Train on combined training and validation sets, and generate test submission. '''

    nav_graphs = setup(args.action_space, args.navigable_locs_path)

    # Create a batch training environment that will also preprocess text

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    feature_store = Feature(features, args.panoramic)

    if args.encoder_path != "":
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))

    encoder.eval()
    decoder.eval()
    # Generate test submission
    test_env = R2RBatch(feature_store, nav_graphs, args.panoramic, args.action_space,batch_size=args.batch_size, splits=['test'], tokenizer=tok,
                        path_type=path_type, history=history, blind=blind)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len, path_type=args.path_type,args=args)
    agent.results_path = '%s%s_%s.json' % (RESULT_DIR, "Submit", 'test')
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()

# NOTE: only available to us, now, for writing the paper.
def train_test(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind):
    ''' Train on the training set, and validate on the test split. '''

    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok, path_type=path_type, history=history, blind=blind),
                        Evaluation([split], path_type=path_type)) for split in ['test']}

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, hidden_size,dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH,
          model_prefix, val_envs=val_envs)

def train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind, args):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    nav_graphs = setup(args.action_space, args.navigable_locs_path)
    #set_trace()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    #train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok,
    #                     path_type=path_type, history=history, blind=blind)

    feature_store = Feature(features, args.panoramic)
    train_env = R2RBatch(feature_store, nav_graphs, args.panoramic,args.action_space,batch_size=args.batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments

    val_envs = {split: (R2RBatch(feature_store,nav_graphs, args.panoramic, args.action_space,batch_size=args.batch_size, splits=[split],
                tokenizer=tok, path_type=path_type, history=history, blind=blind),
                Evaluation([split], path_type=path_type)) for split in ['train','val_seen','val_unseen']}

    train(train_env, tok, n_iters,  path_type, history, max_episode_len, MAX_INPUT_LENGTH, model_prefix, val_envs=val_envs)


if __name__ == "__main__":
    
    blind = args.blind

    # Set default args.
    path_type = args.path_type
    # In MP, max_episode_len = 20 while average hop range [4, 7], e.g. ~3x max.
    # max_episode_len has to account for turns; this heuristically allowed for about 1 turn per hop.
    if path_type == 'planner_path':
        max_episode_len = 20  # [1, 6], e.g., ~3x max
    else:
        #max_episode_len = 80  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]
        max_episode_len = 40  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]

    # Input settings.
    history = args.history
    # In MP, MAX_INPUT_LEN = 80 while average utt len is 29, e.g., a bit less than 3x avg.
    if history == 'none':
        MAX_INPUT_LENGTH = 1  # [<EOS>] fixed length.
    elif history == 'target':
        MAX_INPUT_LENGTH = 3  # [<TAR> target <EOS>] fixed length.
    elif history == 'oracle_ans':
        MAX_INPUT_LENGTH = 70  # 16.16+/-9.67 ora utt len, 35.5 at x2 stddevs. 71 is double that.
    elif history == 'nav_q_oracle_ans':
        #MAX_INPUT_LENGTH = 120  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
        MAX_INPUT_LENGTH = 80  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
    else:  # i.e., 'all'
        MAX_INPUT_LENGTH = 120 * 6  # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A. 5.67 at x2 std. Call it 6 (real max 13).
        #MAX_INPUT_LENGTH = 120 ##300

    # Training settings.
#    feedback_method = args.feedback
    n_iters = args.n_iters

    # Model prefix to uniquely id this instance.
    model_prefix = '%s-seq2seq-%s-%s-%d-imagenet' % (args.eval_type, history, path_type, max_episode_len)
    if blind:
        model_prefix += '-blind'

    if args.eval_type == 'val':
        train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind, args)
    elif args.eval_type == 'traintest':
        train_test(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind)
    else:
        test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, n_iters, model_prefix, blind, args)


    # test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)
