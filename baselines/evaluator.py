#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import copy
import logging
import os
from functools import reduce
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


def cal_gamma_return_of_an_episode(reward_list, gamma):
    n = len(reward_list)
    gamma_list = np.array([np.power(gamma, i) for i in range(n)])
    reward_list = np.array(reward_list)
    gamma_return = np.array([sum(reward_list[i:] * gamma_list[:(n - i)]) for i in range(n)])
    return gamma_return


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, env, model, logdir):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.env = env
        self.model = model
        self.iteration = 0
        self.log_dir = logdir
        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}

    def get_stats(self):
        self.stats.update()
        return self.stats

    def run_an_episode(self, mode):
        reward_list = []
        eval_v_list = []
        info_dict = dict()
        done = 0
        obs = self.env.reset()
        while not done:
            obs = self.tf.constant(obs)
            if mode == 'Performance':
                action = self.model.mode(obs)
                _, eval_v, _, _ = self.model.step(obs)
            else:
                action, eval_v, _, _ = self.model.step(obs)
            eval_v_list.append(eval_v.numpy())
            obs, reward, done, info = self.env.step(action.numpy())
            reward_list.append(reward[0])
        if mode == 'Performance':
            episode_return = sum(reward_list)
            episode_len = len(reward_list)
            info_dict = dict(episode_return=episode_return,
                             episode_len=episode_len)
        elif mode == 'Evaluation':
            true_v_list = list(cal_gamma_return_of_an_episode(reward_list, 0.99))
            info_dict = dict(true_v_list=true_v_list,
                             eval_v_list=eval_v_list)
        return info_dict

    def average_max_n(self, list_for_average, n=None):
        if n is None:
            return sum(list_for_average) / len(list_for_average)
        else:
            sorted_list = sorted(list_for_average, reverse=True)
            return sum(sorted_list[:n]) / n

    def run_n_episodes(self, n, mode):
        epinfo_list = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            episode_info = self.run_an_episode(mode)
            epinfo_list.append(episode_info)
        if mode == 'Performance':
            n_episode_return_list = [epinfo['episode_return'] for epinfo in epinfo_list]
            n_episode_len_list = [epinfo['episode_len'] for epinfo in epinfo_list]
            average_return_with_diff_base = np.array([self.average_max_n(n_episode_return_list, x) for x in [1, 3, 5]])
            average_len = self.average_max_n(n_episode_len_list)
            return dict(average_len=average_len,
                        average_return_with_max1=average_return_with_diff_base[0],
                        average_return_with_max3=average_return_with_diff_base[1],
                        average_return_with_max5=average_return_with_diff_base[2],)
        elif mode == 'Evaluation':
            n_episode_true_v_list = [epinfo['true_v_list'] for epinfo in epinfo_list]
            n_episode_eval_v_list = [epinfo['eval_v_list'] for epinfo in epinfo_list]
            def concat_interest_epi_part_of_one_ite_and_mean(list_of_n_epi, max_state=200):
                tmp = list(copy.deepcopy(list_of_n_epi))
                tmp[0] = tmp[0] if len(tmp[0]) <= max_state else tmp[0][:max_state]

                def reduce_fuc(a, b):
                    return np.concatenate([a, b]) if len(b) < max_state else np.concatenate([a, b[:max_state]])

                interest_epi_part_of_one_ite = reduce(reduce_fuc, tmp)
                return np.mean(interest_epi_part_of_one_ite)
            true_v_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_true_v_list))
            eval_v_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_eval_v_list))
            return dict(true_v_mean=true_v_mean*0.2,
                        eval_v_mean=eval_v_mean*0.2)

    def run_evaluation(self, iteration):
        self.iteration = iteration
        mean_metric_dict = self.run_n_episodes(5, 'Performance')
        mean_metric_dict1 = self.run_n_episodes(5, 'Evaluation')
        mean_metric_dict.update(mean_metric_dict1)
        with self.writer.as_default():
            for key, val in mean_metric_dict.items():
                self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
            self.writer.flush()


if __name__ == '__main__':
    pass
