''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import model_PREVALENT
from utils import padding_idx, print_progress
import pdb
from ipdb import set_trace
from param import args
from collections import defaultdict
import utils
import sys
sys.setrecursionlimit(3000)
class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'inst_idx': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    def vl_rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    def get_results(self):
        #set_trace()
        output = [{'instr_id': k, 'trajectory': v[0], 'path_id': v[1]} for k, v in self.results.items()]
        return output
    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break



class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        # for t in range(30):  # 20 ep len + 10 (as in MP); planner paths
        for t in range(130):  # 120 ep len + 10 (emulating above); player paths
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    """
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']
    """
    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, args=None):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        #dset_trace()
        from param import args
        import utils
        self.path_type = args.path_type
        self.tok = tok
        self.episode_len = 20
        self.feature_size = 2048
        #set_trace()
#        device_ids = [1, 2]


        self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
##        self.vln_bert = torch.nn.DataParallel(self.vln_bert).cuda()

        self.critic = model_PREVALENT.Critic().cuda()
##        self.critic= torch.nn.DataParallel(self.critic).cuda()

        self.models = (self.vln_bert, self.critic)

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)

#        self.vln_bert_optimizer = nn.DataParallel(self.vln_bert_optimizer)
#        self.critic_optimizer = nn.DataParallel(self.critic_optimizer)

        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
#        self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''
        #set_trace()
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx)

    def original_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        #feature_size = obs[0]['feature'].shape[0]
        #features = np.empty((len(obs),feature_size), dtype=np.float32)
        if isinstance(obs[0]['feature'],tuple): # todo?
            features_pano = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][0], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][1], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            for i,ob in enumerate(obs):
                features_pano[i] = ob['feature'][0]
                features[i] = ob['feature'][1]
            return (Variable(torch.from_numpy(features_pano), requires_grad=False).cuda(),
            Variable(torch.from_numpy(features), requires_grad=False).cuda())
        else:
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'], dtype=np.float32),0),len(obs),axis=0)  # jolin
            for i,ob in enumerate(obs):
                features[i] = ob['feature']
            return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        #set_trace()
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32) #(8, 10, 2176)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        #set_trace()
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def _teacher_action_cvdn(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        #set_trace()
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def pteacher_action(self,model_actions, action_space, obs, ended, ignore_index):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            if action_space==6:
                ix,heading_chg,elevation_chg = ob['teacher']
                if heading_chg > 0:
                    a[i] = model_actions.index('right')
                elif heading_chg < 0:
                    a[i] = model_actions.index('left')
                elif elevation_chg > 0:
                    a[i] = model_actions.index('up')
                elif elevation_chg < 0:
                    a[i] = model_actions.index('down')
                elif ix > 0:
                    a[i] = model_actions.index('forward')
                elif ended[i]:
                    a[i] = model_actions.index('<ignore>')
                else:
                    a[i] = model_actions.index('<end>')
            else:
                a[i] = ob['teacher'] if not ended[i] else ignore_index
        return a

#    def _teacher_action(self, obs, ended):
#        a = self.pteacher_action(self.model_actions, self.decoder.action_space, obs, ended, self.ignore_index)
#        return Variable(a, requires_grad=False).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        #set_trace()
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

                state = self.env.env.sims[idx].getState()
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj



    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros((len(obs), max_num_a, action_embedding_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            action_embeddings[i, :num_a, :] = ob['action_embedding'] #bug: todo
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (Variable(torch.from_numpy(action_embeddings), requires_grad=False).cuda(),
                Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
                is_valid)

    def bert_rollout(self):
        world_states = np.array(self.env.reset(False))
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        #ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env


        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs

            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            h_t, c_t, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def vl_rollout(self, train_ml=None, train_rl=True, reset=True):
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env  True
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        train_ml = 0.2

        # Reorder the language input for the encoder

        # Language input
        #language_attention_mask is mask like [1,1,1,1,0,0,0]
        #set_trace()
        sentence, language_attention_mask, token_type_ids, \
            seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]  #sorted

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,  #bs 300
                        'attention_mask': language_attention_mask, #bs,233 
                        'lang_mask':      language_attention_mask, #bs,80
                        'token_type_ids': token_type_ids}  #zero

        h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        #ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)

        # Initial action
        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        
        visited = [set() for _ in range(batch_size)]
        # Initialization the tracking state
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # Do a sequence rollout and calculate the loss
#        self.loss = 0
#        env_action = [None] * batch_size
        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            #change the CLS with h_t(initial vlnbert)
            if (t >= 1) or (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            #visual_temp_mask is visual candidate mask (1,0)
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long() #bs,10 (true, false)
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)

            ##self.vln_bert.module.vln_bert.config.directions = max(candidate_leng)

            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids, #0
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat}
            h_t, logit = self.vln_bert(**visual_inputs)  
          
            hidden_states.append(h_t)


            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]            # Perm the obs for the resu

            if train_rl: #False #True
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
#                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                                reward[i] = 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
#                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                #reward[i] = 1.0 + ndtw_reward
                                reward[i] = 1.0 
                            elif reward[i] < 0.0:
                                #reward[i] = -1.0 + ndtw_reward
                                reward[i] = -1.0 
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl: #False True
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ##self.vln_bert.module.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat}
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                #set_trace()
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    if batch_size == 1:
                        discount_reward[i] = last_value__
                    else:
                        discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())
        
        #set_trace()
        if train_ml is not None: #True
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def vl_rollout_test(self, train_ml=None, train_rl=True, reset=True):
        #set_trace()
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False
        #set_trace()
        if reset:  # Reset env  True
            obs = np.array(self.env.reset_test())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        train_ml = 0.2

        sentence, language_attention_mask, token_type_ids, \
            seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]  #sorted

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,  #bs 300
                        'attention_mask': language_attention_mask, #bs,233 
                        'lang_mask':      language_attention_mask, #bs,80
                        'token_type_ids': token_type_ids}  #zero

        h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        #set_trace()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        #ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)

        # Initial action
        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        
        visited = [set() for _ in range(batch_size)]
        # Initialization the tracking state
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # Do a sequence rollout and calculate the loss
#        self.loss = 0
#        env_action = [None] * batch_size
        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            #change the CLS with h_t(initial vlnbert)
            if (t >= 1) or (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            #visual_temp_mask is visual candidate mask (1,0)
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long() #bs,10 (true, false)
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids, #0
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat}
            h_t, logit = self.vln_bert(**visual_inputs)  #[bs,768] [bs,10]
            #set_trace()
            hidden_states.append(h_t)


            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]            # Perm the obs for the resu

            
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break


        return traj

    def test(self, use_dropout=False, feedback='argmax', iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        #set_trace()
#        if not allow_cheat: # permitted for purpose of calculating validation loss only
#            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
#        if use_dropout:
#            self.vln_bert.train()
#            self.critic.train()
 #       else:
        self.vln_bert.eval()
        self.critic.eval()

        self.owntest()

    def owntest(self, **kwargs):
        #set_trace()
        iters= 114
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        #self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.instr_idsall = []
        #set_trace()
        for i in range(iters):
            for traj in self.vl_rollout_test():
                if traj['inst_idx'] not in self.instr_idsall:
                    self.loss = 0
                    self.results[traj['inst_idx']] = traj['path']
                    self.instr_idsall.append(traj['inst_idx'])

    def test_sub(self, use_dropout=False, feedback='argmax', iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        #set_trace()
#        if not allow_cheat: # permitted for purpose of calculating validation loss only
#            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()

        self.owntest_sub()

    def owntest_sub(self):
        iters= 173
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        #self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.instr_idsall = []
        #set_trace()
        for i in range(iters):
            for traj in self.vl_rollout_test():
                if traj['inst_idx'] not in self.instr_idsall:
                    self.loss = 0
                    self.results[traj['inst_idx']] = traj['path']
                    self.instr_idsall.append(traj['inst_idx'])

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()


    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        #set_trace()
        from param import args

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.vl_rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.vl_rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.vl_rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()
            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)


    def save_cvdn(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_cvdn(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
#        all_tuple = [("vln_bert", self.vln_bert.module, self.vln_bert_optimizer),
#                     ("critic", self.critic.module, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1




