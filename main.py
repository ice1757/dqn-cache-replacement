import random
import sys
import time
# from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app
from absl import flags
from absl import logging

from agents.Agent import *
# from agents.model_AC import AC
# from agents.model_DRQN import DRQN
# from agents.model_DDPG import DDPG
from agents.model_DQN import DQN
from ENV.new_Cache_test import Cache
from Func import Temp_Memory
# from func_draw import plot_, plot_multi
from func_loader import save_immediate_reward, check_folder, save_long_term_reward, load_parameter, save_hit_rate, cnt_diff_hit_rate, savetxt, savetxt_look, jwrt

random.seed(1234)

FLAGS = flags.FLAGS
## define the default value
flags.DEFINE_integer('cache_size', 5, 'set cache_size', lower_bound=0)
flags.DEFINE_string('experiment_name', 't1', 'the experiment name')
flags.DEFINE_string('dataset', 'paper8_data_orig', 'the dataset stored in the folder ./dataset')
flags.DEFINE_string('req_to_size_dataset', 'gen1', 'the requests correspond to its size stored in the folder ./dataset/userdata')
flags.DEFINE_string('model_filepath', None, 'the filepath of model parameters')
flags.DEFINE_string('feature_filepath', './dataset/parameter/features/Ffqr_norm/full_norm.json', 'the filepath of feature combinations')

def main(argv):
    cache_size = FLAGS.cache_size
    experiment_name = FLAGS.experiment_name
    file_name = FLAGS.dataset
    req_size_file_name = FLAGS.req_to_size_dataset
    skip_hit = True
    skip_evict = False
    record_reward = True
    record_cost = True
    assist_learn = [LFU(), LRU()] 					# None, LRU(), LFU(),...
    # assist_learn = None
    normalize_reward = True					# you need to set this, if delay reward == True # unused
    upset_memory = False
    if FLAGS.model_filepath:
        model_name = FLAGS.model_filepath.split('/')[-1].split('.')[0]
        feature_name = FLAGS.feature_filepath.split('/')[-1].split('.')[0]
        ## load model parameter
        model_parameter = load_parameter(FLAGS.model_filepath)
        ## setting DQN parameter
        train_episodes = model_parameter["DQN"]["train_episodes"]
        train_step = model_parameter["DQN"]["train_step"]
        start_train_step = model_parameter["DQN"]["start_train_step"]
        learning_rate = model_parameter["DQN"]["learning_rate"]
        discount_factor = model_parameter["DQN"]["discount_factor"]
        e_greedy = model_parameter["DQN"]["e_greedy"]
        replace_target_iter = model_parameter["DQN"]["replace_target_iter"]
        memory_size = model_parameter["DQN"]["memory_size"]
        batch_size = model_parameter["DQN"]["batch_size"]
        e_greedy_increment = model_parameter["DQN"]["e_greedy_increment"]
        layers = model_parameter["DQN"]["layers"]
        ## setting penalty_parameter
        penalty_parameter = model_parameter["penalty_parameter"]
        ## setting reward function
        reward_method = None if model_parameter["reward_function"] == 'our' else model_parameter["reward_function"]	# 'pre_len_rank', 'pre_len', 'hit_change', 'hit_rate', 'freq^-1', '1-past_freq_l', 'wang', 'zhong',None
        delay_reward = True if reward_method == None else False
        logging.info("reward_method: %s, delay_reward: %s", reward_method, delay_reward)
    else:
        model_name='tradition'
        feature_name='none'
        reward_method = None
        penalty_parameter = 0
    
    save_hit = False
    show_action_history = False

    ## load feature parameter
    feature_parameter = load_parameter(FLAGS.feature_filepath)
    ## setting DQN parameter
    need_feature = feature_parameter["need_feature"]
    new_req_feature = feature_parameter["new_req_feature"]
    

    env = Cache(
        file_name=file_name,
        cache_size=cache_size, 
        req_size_file_name = req_size_file_name,
        # feature
        need_feature=need_feature,
        # now request feature
        new_req_feature=new_req_feature,
        skip_hit=skip_hit,
        skip_evict=skip_evict,
        reward_method=reward_method,
        penalty_parameter=penalty_parameter
    )

    ## 數 json 檔裡面有多少個 true 即多少 feature
    need_feature_len = sum(value == True for value in need_feature.values())
    if need_feature['Far']: need_feature_len += (env.onehot_len - 1)			# Previous step is already plus 1 for "tg"

    new_req_feature_len = sum(value == True for value in new_req_feature.values())
    if new_req_feature['Fcr']: new_req_feature_len += (env.onehot_len - 1)	# Previous step is already plus 1 for "tg"

    agents = dict()
    peep_ = belady()
    save_path = "./logs/"+experiment_name+"/"+file_name+"_c"+str(cache_size)+"_"+req_size_file_name+"/"+model_name+'_'+feature_name
    if not FLAGS.model_filepath:
        agents['FIFO'] = FIFO()
        agents['LRU']  = LRU()
        agents['LFU']  = LFU()
        agents['random']  = Random()
        agents['GDS'] = "GDS"
        agents['belady']  = belady()
        agents['belady_size'] = belady_size()
        
    else:
        n_features = need_feature_len * cache_size + new_req_feature_len
        n_actions = cache_size if not skip_evict else cache_size + 1
        TM = Temp_Memory(memory_size=memory_size, n_features=n_features)
        agents['DQN'] = DQN(n_actions, n_features,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            e_greedy=e_greedy,
            replace_target_iter=replace_target_iter,
            memory_size=memory_size,
            batch_size=batch_size,
            e_greedy_increment=e_greedy_increment,
            assist_learn=assist_learn,
            upset_memory=upset_memory,
            normalize_reward=normalize_reward,
            layers=layers
        )
    logging.info("Agent: %s", agents)
    # heu_run = 1
    start_time = time.time()
    
    if save_hit or record_reward: 
        if not check_folder(save_path): raise 'check_folder error 1'
    jwrt(f'{save_path}_f', feature_parameter)
    if FLAGS.model_filepath:
        jwrt(f'{save_path}_f', model_parameter)
    env.show_cache()

    for method, agent in agents.items():
        if method == "GDS" : heu_run = 5 
        else : heu_run = 1
        episodes = train_episodes if method == "DQN"  else heu_run
        hit_rates = []
        if record_reward: 
            episode_ireward = [] # immediate
            episode_lreward = [] # long-term Q(s,a)
        if record_cost: episode_cost = []
        for episode in range(episodes):
            print("-----------Episode: %s-----------" % (episode+1))
            s = env.reset()
            print("---env initial---")
            done = False
            step = 0
            peep_acc = 0

            if show_action_history: action_history = []
            reward_history = []
            if method == "DQN":
                TM.reset()
                # agent.mem_reset()
            # savetxt_look('cache_block', f'{step}')
            with tqdm.tqdm(total=env.requests_len, disable=False) as pbar:
                while not done:
                    # savetxt(f'./look/cache_block', f'{env.new_request}, {env.cache_blocks}, {env.cache_freq}')
                    while True:
                        new_cache = env.new_request
                        # peep_a = peep_.choose_action(s)
                        if method == "DQN": 
                            a = agent.choose_action(s)
                        elif method == "GDS":
                            a = env.gds_sc()
                            break
                        else:
                            a = agent.choose_action(s)
                        # savetxt_look(f'cache_block', f'{env.new_request}, {env.cache_blocks}, {a}, {env.cache_freq}')
                        if a < len(env.cache_blocks): 
                            victim = env.cache_blocks[a]
                            break # valid action
                        elif skip_evict and a <= len(env.cache_blocks): 
                            # when skip_evict==True, "a" may be equal to the size of cache_blocks
                            break
                    # r = s['next_req'][a] - env.data_pointer # reuse distance
                    if method == "GDS":
                        for se_a in a:
                            if env.data_pointer == env.requests_len:
                                break
                            idx = env.cache_blocks.index(se_a)
                            s_, r, done, info = env.step(idx)
                    else:    
                        s_, r, done, info = env.step(a)
                    if show_action_history: action_history.append(a)
                    # r = 10 if peep_a == a else -10
                    # if r == 10: peep_acc += 1
                    if method == "DQN" :
                        if not skip_evict:
                            if not delay_reward: 
                                agent.store_transition(s['feature'], a, r, s_['feature'])
                                if record_reward: reward_history.append(r)
                            else: 
                                pre_transitions = TM.store_temp_transition(s['feature'], a, r, s_['feature'], new_cache, victim, s['c_step'], 
                                                                           env.cache_blocks, env.cache_freq, env.requests[:s['c_step']])
                                if pre_transitions:
                                    for Ps, Pa, Pr, Ps_ in pre_transitions:
                                        agent.store_transition(Ps, Pa, Pr, Ps_)
                                        if record_reward: reward_history.append(Pr)
                        else: # can skip evict, store transition directly
                            agent.store_transition(s['feature'], a, r, s_['feature'])
                        if (step > start_train_step) and (step % train_step == 0): agent.learn()
                        # if step % 1000 == 0:
                        # 	env.pretty_print_hitrate()
                    else:
                        pass
                    pbar.update(s_['c_step'] - s['c_step'])
                    s = s_
                    step += 1
                    
            hit_rates.append(env.hit_history.count(1) / len(env.hit_history))
            # print('best choose', peep_acc / step)
            print("E: ", method, len(env.hit_history), "hit rate:", hit_rates[-1])
            # print("S: ", cnt_diff_hit_rate(env.hit_history, 3))
            if method == "DQN" and skip_evict == True:
                print("skip evict rate:", env.skip_evict_time / env.hit_history.count(0))
            if save_hit: env.save_hit_history(method, save_path)
            if show_action_history:
                for i in range(max(action_history)+1):
                    print(i, action_history.count(i))
            if record_reward and method == "DQN":
                # immediate reward
                episode_ireward.append(np.mean(reward_history))
                # long-term reward
                episode_lreward.append(np.mean(agent.reward_his))
                agent.reward_his = []
            if record_cost and method == "DQN":
                episode_cost.append(np.mean(agent.cost_his))
                agent.cost_his = []
            save_hit_rate(save_path, hit_rates, method+'_'+feature_name+'_hitrate')
        end_time = time.time()
        print("time :", end_time-start_time)
        print("avg hit rate :", sum(hit_rates) / len(hit_rates))
        print("max hit rate :", max(hit_rates))
        print("min hit rate :", min(hit_rates))
        savetxt(f'{save_path}_hit', f'{method}: {round(sum(hit_rates) / len(hit_rates), 4)}, {round(min(hit_rates), 4)}, {round(max(hit_rates), 4)}')
        if save_hit: save_hit_rate(save_path, hit_rates, model_name+'_'+feature_name)
        if record_reward: 
            save_immediate_reward(save_path, episode_ireward, model_name+'_'+feature_name)
            save_long_term_reward(save_path, episode_lreward, model_name+'_'+feature_name)	
if __name__ == '__main__':
    app.run(main)