import logging
from math import sqrt

class ServerConfig:
    mode = "train"  # under train mode, model will update params, and pick actions base on the softmax output probabilities;
                    # under test mode, model will not update params, and pick the action with most chance;
                    # all models are saved/loaded at PPOConfig.save_name

class PPOConfig:
    gamma = 1  # discount rate
    epsilon = 0.25  # clip
    entropy_coef = 0.05 # entropy weight
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-2
    latency_weight = 1
    fail_punish = -sqrt(300)
    shuffle_punishment = -0.1
    use_shaping_reward = True
    lmbda = 0.95
    epochs = 10
    optimizer = "Adam"
    device = 'cuda'  
    save_name = "lqrs_job"  # models params are saved in models/checkpoints
    save_interval = 50

# identify the database and the table file. You can add your custom databases, but remember to make a table_file like imdb_tables.csv
class EnvironmentConfig:
    database = "imdb"
    table_file = f'/home/xxxxxxxx/LQRS_release/benchmarks/{database}_tables.csv'
    if database == "tpch":
        table_num = 8
        col_num = 16
    if database == "imdb":
        table_num = 17
        col_num = 34
    if database == "stack":
        table_num = 11
        col_num = 25

class LoggingConfig:
    log_level = logging.INFO
