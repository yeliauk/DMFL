import argparse

# from experiments.runner import ExperimentRunner
# from ..runner import ExperimentRunner

import sys
sys.path.append('E:\\PyCharmProjects\\DMFL')  # 使得该项目目录能被识别，下面导入runner时能正常导入
                                               # 防止目录中的\+字母被识别为转义字符  解决办法1：目录名前面加r；解决办法2：用\\代替\；解决办法3：用/代替\
from experiments.runner import ExperimentRunner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on the MNIST dataset using DMFL.")
    parser.add_argument(
        '--full',
        help='Do a full run. Otherwise, does a quick run for testing purposes during development.',
        action='store_true'
    )
    args = parser.parse_args()

    QUICK_RUN = not args.full  # 是否快速run
    if QUICK_RUN:  #设置训练迭代次数
        TRAINING_ITERATIONS = 1
    else:
        TRAINING_ITERATIONS = 50
    # TRAINING_HYPERPARAMS = {  # 训练超参数  MNIST
    #     'final_round_num': TRAINING_ITERATIONS,
    #     'batch_size': 32,
    #     'epochs': 1,
    #     'learning_rate': 1e-2
    # }
    TRAINING_HYPERPARAMS = {  # 训练超参数  用电量
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 32,
        'epochs': 4,
        'learning_rate': 1e-2
    }
    DP_PARAMS = [  # 与dp有关的参数
        {
            'l2_norm_clip': 1.0,
            'noise_multiplier': 0.9,
            'delta': 1e-5
        },
        {
            'l2_norm_clip': 1.0,
            'noise_multiplier': 1.3,
            'delta': 1e-5
        },
        {
            'l2_norm_clip': 1.0,
            'noise_multiplier': 1.5,
            'delta': 1e-5
        },
    ]
    ROUND_DURATION = 5400  # should always end early  每轮训练应该总是在该段时间内结束
    # ROUND_DURATION = 3600  # 时间设置长一些
 
    runner = ExperimentRunner(  # runner实例化
        QUICK_RUN,
        TRAINING_ITERATIONS,
        TRAINING_HYPERPARAMS,
        ROUND_DURATION
    )

    if QUICK_RUN:  # 如果是快速run
        # experiments = []  # 拆分数据，没有要求  这是原来代码的写法，需要填入参数
        #experiments = [{'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 3}]  # 平均拆分测试数据，给各个trainer
        # 加入合同理论
        experiments = [{'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 3,
                        'clients_num': 3, 'clients_data_quality': {'Bob':2, 'Carol':2, 'David':2} }]
        method = 'step'  # 评估方法为step
        seed = 88  # 随机种子
        for exp in experiments:
            protocol = 'DMFL'
            runner.run_experiment(protocol=protocol,  # 开始运行
                           eval_method=method, seed=seed, **exp)
    else:  # 如果不是快速run
        experiments = [  # 设置拆分数据的多种要求

            # Test 平均分
            #{'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 3},

            # 加入合同理论
            # iid
            # {'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 3,  # test
            #  'clients_num': 3, 'clients_data_quality': {'Bob':2, 'Carol':2, 'David':2}},

            # {'dataset': 'mnist', 'split_type': 'size', 'num_trainers': 3, 'ratios': [1, 2, 7],  # acc: 0.9756(IC)   0.9759(I)  0.9705  model line[0.32, 0.50, 0.85]
            #  'clients_num': 3, 'clients_data_quality': {'Bob': 1, 'Carol': 2, 'David': 4}},

            # {'dataset': 'mnist', 'split_type': 'size', 'num_trainers': 3, 'ratios': [1, 3, 6],  # acc: 0.9734(IC)  0.9736(I)  0.9689  model line[0.32, 0.70, 0.80]
            #  'clients_num': 3, 'clients_data_quality': {'Bob': 1, 'Carol': 5, 'David': 6}},

            # {'dataset': 'mnist', 'split_type': 'size', 'num_trainers': 4, 'ratios': [1, 2, 3, 4],  # acc: 0.9611(IC)  0.9612(I)  0.9584  model line[0.35, 0.50, 0.65, 0.80]
            #  'clients_num': 4, 'clients_data_quality': {'Bob': 1, 'Carol': 2, 'David': 3, 'Eve': 4}},

            # {'dataset': 'mnist', 'split_type': 'size', 'num_trainers': 5, 'ratios': [1, 2, 3, 4, 5],  # acc: 0.9559(IC)  0.9560(I)  0.9490    model line[0.15, 0.50, 0.60, 0.70, 0.75]
            # 'clients_num': 5, 'clients_data_quality': {'Bob': 3, 'Carol': 10, 'David': 12, 'Eve': 14, 'Frank':15}},

            # non-iid
            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 3, 'ratios': [1, 2, 7],
            #  'disjointness': 0.6,  # acc: 0.9638(IC)   0.9640(I)  0.9519  model line[0.15, 0.20, 0.80]
            #  'clients_num': 3, 'clients_data_quality': {'Bob': 3, 'Carol': 4, 'David': 16}
            # },

            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 3, 'ratios': [1, 3, 6],
            #  'disjointness': 0.6,  # acc: 0.9564(IC)   0.9565(I)  0.9501  model line[0.15, 0.55, 0.75]
            #  'clients_num': 3, 'clients_data_quality': {'Bob': 3, 'Carol': 11, 'David': 15}
            # },

            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 5, 'ratios': [1, 2, 3, 4, 5],
            #  'disjointness': 0.6,  # acc: 0.9371(IC)   0.9373(I)  0.9244  model line[0.05, 0.15, 0.20, 0.55, 0.60]
            #  'clients_num': 5, 'clients_data_quality': {'Bob': 1, 'Carol': 3, 'David': 4, 'Eve': 11, 'Frank':12}
            # }

            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 9,
            #  'ratios': [3,4,5,7,10,14,19,26,32],
            #  'disjointness': 0.8,  # acc:  (IC)   0.8504(I)  0.8351  model line[0.04, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1]
            #  'clients_num': 9,
            #  'clients_data_quality': {'Bob': 8, 'Carol': 10, 'David': 12, 'Eve': 14, 'Frank': 16,
            #                           'Georgia': 17, 'Henry': 18, 'Isabel': 19, 'Joe': 20}
            # }


            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 10,
            #  'ratios': [2, 3, 4, 5, 7, 10, 13, 17, 24, 35],
            #  'disjointness': 0.8,
            #  # acc: 0.(IC)   0.8623(I)  0.8455   model line[0.0230, 0.0250, 0.0270, 0.0290, 0.0310, 0.0330, 0.0350, 0.0370, 0.0390, 0.0410]
            #  'clients_num': 10,
            #  'clients_data_quality': {'Bob': 23, 'Carol': 25, 'David': 27, 'Eve': 29, 'Frank': 31,
            #                           'Georgia': 33, 'Henry': 35, 'Isabel': 37, 'Joe': 39, 'Kelly': 41}
            # }

            # {'dataset': 'mnist', 'split_type': 'size+noniid', 'num_trainers': 10,
            #  'ratios': [2, 3, 4, 5, 7, 10, 13, 17, 24, 35],
            #  'disjointness': 0.9,
            #  # acc: 0.(IC)   0.(I)  0.5808   model line[0.0230, 0.0250, 0.0270, 0.0290, 0.0310, 0.0330, 0.0350, 0.0370, 0.0390, 0.0410]
            #  'clients_num': 10,
            #  'clients_data_quality': {'Bob': 23, 'Carol': 25, 'David': 27, 'Eve': 29, 'Frank': 31,
            #                           'Georgia': 33, 'Henry': 35, 'Isabel': 37, 'Joe': 39, 'Kelly': 41}
            #  }


            # 用电量数据
            {'dataset': 'electricity', 'num_trainers': 9, 'split_type': 'no_action',
             'clients_num': 9,
             'clients_data_quality': {'Bob': 1, 'Carol': 1, 'David': 1, 'Eve': 1, 'Frank': 1,
                                      'Georgia': 1, 'Henry': 1, 'Isabel': 1, 'Joe': 1}
            }

            # {'dataset': 'electricity', 'num_trainers': 3, 'split_type': 'no_action',
            #  'clients_num': 3,
            #  'clients_data_quality': {'Bob': 1, 'Carol': 1, 'David': 1}
            # }

        ]
        method = 'step'  # 评估方法为step 
        seed = 89  # 随机种子
        for exp in experiments:  # 在不同拆分数据情况下进行实验
            protocol = 'DMFL'
            print(f"Starting experiment with args: {exp}")
            runner.run_experiment(protocol=protocol,  # 开始运行
                            eval_method=method, seed=seed, **exp)