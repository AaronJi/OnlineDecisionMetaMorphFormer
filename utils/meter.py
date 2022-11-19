from collections import defaultdict
from collections import deque

import numpy as np

from config import cfg


class AgentMeter:
    def __init__(self, name, env='unimal'):
        self.name = name
        self.env = env

        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count = 0
        self.ep_len = deque(maxlen=10)
        self.ep_len_ema = -1
        self.ema_start_ep_len = 10
        #self.ema_start_ep_len = 2  # TODO

    def add_ep_info(self, infos):
        if self.env == 'unimal':
            for info in infos:
                if info["name"] != self.name:
                    continue
                if "episode" in info.keys():
                    self.ep_rew["reward"].append(info["episode"]["r"])
                    self.ep_count += 1
                    self.ep_len.append(info["episode"]["l"])
                    if self.ep_count == self.ema_start_ep_len:
                        self.ep_len_ema = np.mean(self.ep_len)
                    elif self.ep_count >= self.ema_start_ep_len:
                        alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                        self.ep_len_ema = (alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema)

                    for rew_type, rew_ in info["episode"].items():
                        if "__reward__" in rew_type:
                            self.ep_rew[rew_type].append(rew_)

                    if "x_pos" in info:
                        self.ep_pos.append(info["x_pos"])
                    if "x_vel" in info:
                        self.ep_vel.append(info["x_vel"])
                    if "metric" in info:
                        self.ep_metric.append(info["metric"])
        else:
            for ep_info in infos["episode"]:
                if ep_info is not None:
                    self.ep_rew["reward"].append(ep_info["r"])
                    self.ep_count += 1
                    self.ep_len.append(ep_info["l"])
                    if self.ep_count == self.ema_start_ep_len:
                        self.ep_len_ema = np.mean(self.ep_len)
                    elif self.ep_count >= self.ema_start_ep_len:
                        alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                        self.ep_len_ema = (alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema)

                    if "metric" in ep_info:
                        self.ep_metric.append(ep_info["metric"])



    def update_mean(self):
        if len(self.ep_rew["reward"]) == 0:
            return False

        for rew_type, rews_ in self.ep_rew.items():
            self.mean_ep_rews[rew_type].append(round(np.mean(rews_), 2))

        self.mean_pos.append(round(np.mean(self.ep_pos), 2))
        self.mean_vel.append(round(np.mean(self.ep_vel), 2))
        self.mean_metric.append(round(np.mean(self.ep_metric), 2))
        self.mean_ep_len.append(round(np.mean(self.ep_len), 2))
        return True

    def log_stats(self, max_name_len):
        if len(self.ep_rew["reward"]) == 0:
            return
        ep_rew = self.ep_rew["reward"]
        print(
            "Agent {:>{size}}: mean/median reward {:>4.0f}/{:<4.0f}, "
            "min/max reward {:>4.0f}/{:<4.0f}, "
            "#Ep: {:>7.0f}, avg/ema Ep len: {:>4.0f}/{:>4.0f}".format(
                self.name,
                np.mean(ep_rew),
                np.median(ep_rew),
                np.min(ep_rew),
                np.max(ep_rew),
                self.ep_count,
                np.mean(self.ep_len),
                self.ep_len_ema,
                size=max_name_len
            )
        )


class TrainMeter:
    def __init__(self, env=None):
        if env is None:
            self.env = 'unimal'
        else:
            self.env = env

        if self.env == 'unimal':
            self.agents = cfg.ENV.WALKERS
        else:
            self.agents = ['agent0']

        self.max_name_len = max([len(a) for a in self.agents])

        self.agent_meters = {agent: AgentMeter(agent, env=self.env) for agent in self.agents}

        # Env stats
        self.train_stats = defaultdict(list)
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

    def add_train_stat(self, stat_type, stat_value):
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.add_ep_info(infos)

    def update_mean(self):
        for _, agent_meter in self.agent_meters.items():
            success = agent_meter.update_mean()
            if not success:
                return

        metrics = ["mean_pos", "mean_vel", "mean_metric", "mean_ep_len"]

        for metric in metrics:
            metric_list = []
            for _, agent_meter in self.agent_meters.items():
                metric_list.append(getattr(agent_meter, metric)[-1])

            getattr(self, metric).append(round(np.mean(metric_list), 2))

        rew_types = self.agent_meters[self.agents[0]].mean_ep_rews.keys()

        for rew_type in rew_types:
            rew_list = []
            for _, agent_meter in self.agent_meters.items():
                rew_list.append(agent_meter.mean_ep_rews[rew_type][-1])

            self.mean_ep_rews[rew_type].append(round(np.mean(rew_list), 2))

    def log_stats(self):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.log_stats(self.max_name_len)

        if len(self.mean_ep_rews["reward"]) > 0:
            print("Agent {:>{size}}: mean/------ reward {:>4.0f}, ".format("__env__", self.mean_ep_rews["reward"][-1], size=self.max_name_len))
        return

    def log_global_states(self):
        print('*&'*20)
        rewards_global = 0.0
        len_eps_global = 0.0
        num_eps_global = 0

        for _, agent_meter in self.agent_meters.items():
            print('*&' * 20)
            print(agent_meter.name)
            print(agent_meter.ep_rew)
            print(agent_meter.ep_len)
            print(np.mean(agent_meter.ep_rew["reward"]))
            print(np.mean(agent_meter.ep_len))
            print(agent_meter.ep_count)
            rewards_global += np.mean(agent_meter.ep_rew["reward"])*agent_meter.ep_count
            len_eps_global += np.mean(agent_meter.ep_len)*agent_meter.ep_count
            num_eps_global += agent_meter.ep_count
            exit(6)

        rewards_global /= num_eps_global
        len_eps_global /= num_eps_global
        print("global performance by so far: mean reward %f, mean episode length %f" % (rewards_global, len_eps_global))
        exit(5)
        return


    def get_stats(self):
        stats = {}
        for agent, agent_meter in self.agent_meters.items():
            stats[agent] = {
                "reward": agent_meter.mean_ep_rews,
                "pos": agent_meter.mean_pos,
                "vel": agent_meter.mean_vel,
                "metric": agent_meter.mean_metric,
                "ep_len": agent_meter.mean_ep_len,
            }

        stats["__env__"] = {
                "reward": self.mean_ep_rews,
                "pos": self.mean_pos,
                "vel": self.mean_vel,
                "metric": self.mean_metric,
                "ep_len": self.mean_ep_len,
        }
        stats["__env__"].update(dict(self.train_stats))
        return stats

def log_batch_ep_info(infos, verbal=True):
    performance_ep = {'num': 0, 'reward': 0, 'metric': 0, 'ep_reward': 0, 'ep_len': 0, 'ep_num': 0, 'ep_metric': 0}

    #performance_ep['num'] = len(infos)
    for info in infos:
        if 'metric' in info:
            performance_ep['metric'] += info['metric']
        '''
        x_pos -0.0015536422694050664
        x_vel -0.016537064093403324
        xy_pos_before [-0.0012229   0.00325438]
        xy_pos_after [-0.00155364  0.00443757]
        __reward__forward -0.016537064093403324
        metric -0.0015536422694050664
        name floor-1409-1-4-01-09-49-50
        mj_step_error False
        __reward__ctrl 0.0
        __reward__energy 2256
        __reward__stand 0.0
        '''
        #agent_name = info["name"]
        if "episode" in info.keys():
            performance_ep['ep_reward'] += info["episode"]["r"]
            performance_ep['ep_len'] += info["episode"]["l"]
            performance_ep['ep_num'] += 1

            performance_ep['reward'] += info["episode"]["r"]
            performance_ep['num'] += info["episode"]["l"]
        else:
            continue

    if performance_ep['num'] > 0:
        performance_ep['reward'] /= performance_ep['num']
        performance_ep['metric'] /= performance_ep['num']

    if performance_ep['ep_num'] > 0:
        performance_ep['ep_reward'] /= performance_ep['ep_num']
        performance_ep['ep_len'] /= performance_ep['ep_num']
    if verbal:
        print("current iter performance: mean ep reward %f, mean episode length %f, mean reward %f, mean metric %f, num step %i, num episoe %i" % (performance_ep['ep_reward'], performance_ep['ep_len'], performance_ep['reward'], performance_ep['metric'], performance_ep['num'], performance_ep['ep_num']))
    return performance_ep
