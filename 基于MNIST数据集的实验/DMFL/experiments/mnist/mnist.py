import argparse

# from experiments.runner import ExperimentRunner
# from ..runner import ExperimentRunner

import sys
import os

path = os.path.realpath('..') + '\\DMFL'
sys.path.append(path)

from experiments.runner import ExperimentRunner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="运行MNIST数据集实验")
    parser.add_argument(
        '--full',
        help='完整运行，若不添加为测试运行',
        action='store_true'
    )
    args = parser.parse_args()

    QUICK_RUN = not args.full  # 是否快速run
    if QUICK_RUN:  #设置训练迭代次数
        TRAINING_ITERATIONS = 1
    else:
        TRAINING_ITERATIONS = 5
    TRAINING_HYPERPARAMS = {  # 训练超参数
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 32,
        'epochs': 1,
        'learning_rate': 1e-2
    }
    ROUND_DURATION = 1800  #每轮训练应该总是在该段时间内结束
    # ROUND_DURATION = 3600  # 时间设置长一些
 
    runner = ExperimentRunner(  # runner实例化
        QUICK_RUN,
        TRAINING_ITERATIONS,
        TRAINING_HYPERPARAMS,
        ROUND_DURATION
    )

    if QUICK_RUN:  # 如果是快速run
        experiments = [{'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 3,
                        'clients_num': 3, 'clients_data_quality': {'Bob':2, 'Carol':2, 'David':2} }]
        method = 'step'  # 评估方法为step
        seed = 88  # 随机种子
        for exp in experiments:
            runner.run_experiment(eval_method=method, seed=seed, **exp)
    else:
        experiments = [  # 设置拆分数据的多种要求

            {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 9,
             'ratios': [3,4,5,7,10,14,19,26,32],
             'disjointness': 0.8,
             'clients_num': 9,
             'clients_data_quality': {'Bob': 0.15, 'Carol': 0.20, 'David': 0.25, 'Eve': 0.35, 'Frank': 0.50,   #数据质量与各个参与方本地数据量成正比，比例系数为0.0001
                                      'Georgia': 0.70, 'Henry': 0.95, 'Isabel': 1.3, 'Joe': 1.6},
             'aggregation_model':'reward'  # 奖励比例聚合
            }


            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 10,
            #  'ratios': [2, 3, 4, 5, 7, 10, 13, 17, 24, 35],
            #  'disjointness': 0.8,
            #  'clients_num': 10,
            #  'clients_data_quality': {'Bob': 0.1, 'Carol': 0.15, 'David': 0.2, 'Eve': 0.25, 'Frank': 0.35,
            #                           'Georgia': 0.5, 'Henry': 0.65, 'Isabel': 0.85, 'Joe': 1.2, 'Kelly': 1.75},
            #  'aggregation_model':'reward'
            # }

            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 10,
            #  'ratios': [2, 3, 4, 5, 7, 10, 13, 17, 24, 35],
            #  'disjointness': 0.9,
            #  'clients_num': 10,
            #  'clients_data_quality': {'Bob': 0.1, 'Carol': 0.15, 'David': 0.20, 'Eve': 0.25, 'Frank': 0.35,
            #                           'Georgia': 0.50, 'Henry': 0.65, 'Isabel': 0.85, 'Joe': 1.2, 'Kelly': 1.75},
            #  'aggregation_model': 'reward'
            #  }

        ]
        method = 'step'  # 评估方法为step 
        seed = 89  # 随机种子
        for exp in experiments:
            print(f"Starting experiment with args: {exp}")
            runner.run_experiment(eval_method=method, seed=seed, **exp)
