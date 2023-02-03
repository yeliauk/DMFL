import argparse
import os
import sys
sys.path.append(os.path.realpath('..')+'\\DMFL')
from experiments.runner import ExperimentRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基于行业用电量数据的实验")
    parser.add_argument(
        '--full',
        help='完整运行，如果不加为快速测试模式。',
        action='store_true'
    )
    args = parser.parse_args()

    QUICK_RUN = not args.full  # 是否快速run
    if QUICK_RUN:  #设置训练迭代次数
        TRAINING_ITERATIONS = 1
    else:
        TRAINING_ITERATIONS = 50 #
    TRAINING_HYPERPARAMS = {  # 训练超参数  用电量   !!!!
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 32,
        'epochs': 4,
        'learning_rate': 1e-2
    }
    ROUND_DURATION = 5400  #每轮训练应该总是在该段时间内结束
    # ROUND_DURATION = 3600  # 时间设置长一些
 
    runner = ExperimentRunner(  # runner实例化
        QUICK_RUN,
        TRAINING_ITERATIONS,
        TRAINING_HYPERPARAMS,
        ROUND_DURATION
    )

    if QUICK_RUN:  # 如果是快速run
        # 加入合同理论
        experiments = [{'dataset': 'electricity', 'num_trainers': 9, 'split_type': 'no_action',
                        'clients_num': 9, 'clients_data_quality': {'Bob': 0.93, 'Carol': 0.876, 'David': 0.897, 'Eve': 0.735, 'Frank': 0.806,
                                      'Georgia': 0.794, 'Henry': 0.786, 'Isabel': 0.981, 'Joe': 0.825},
                        'aggregation_model': 'reward'
                        }]
        method = 'step'  # 评估方法为step
        seed = 88  # 随机种子
        for exp in experiments:
            runner.run_experiment(eval_method=method, seed=seed, **exp)
    else:  # 如果不是快速run
        experiments = [  # 设置拆分数据的多种要求
            # 用电量数据
            {'dataset': 'electricity', 'num_trainers': 9, 'split_type': 'no_action',
             'clients_num': 9,
             # 数据质量可看做各个参与方自主计算上传 计算可使用calculateDataQuality.py计算，下面的值即为计算结果
             'clients_data_quality': {'Bob': 0.93, 'Carol': 0.876, 'David': 0.897, 'Eve': 0.735, 'Frank': 0.806,
                                      'Georgia': 0.794, 'Henry': 0.786, 'Isabel': 0.981, 'Joe': 0.825},
             'aggregation_model': 'reward'  # 奖励比例聚合
            }


            # {'dataset': 'electricity', 'num_trainers': 50, 'split_type': 'no_action',
            #  'clients_num': 50,
            #  # 数据质量可看做各个参与方自主计算上传 计算可使用calculateDataQuality.py计算，下面的值即为计算结果
            #  'clients_data_quality': {'Bob': 0.904, 'Carol': 0.961, 'David': 0.896, 'Eve': 0.872, 'Frank': 0.895,
            #                           'Georgia': 0.731, 'Henry': 0.93, 'Isabel': 0.885, 'Joe': 0.521, 'kelly':0.935,
            #                           'B2':0.89, 'C2':0.909, 'D2':0.999, 'E2':0.330, 'F2':0.898, 'G2':0.894, 'H2':0.986, 'I2':0.943, 'J2':0.875, 'K2':0.883,
            #                           'B3':0.781, 'C3':0.919, 'D3':0.971, 'E3':0.911, 'F3':0.88, 'G3':0.508, 'H3':0.871, 'I3':0.755, 'J3':0.912, 'K3':0.98,
            #                           'B4':0.9, 'C4':0.901, 'D4':0.699, 'E4':0.928, 'F4':0.908, 'G4':0.877, 'H4':0.909, 'I4':0.896, 'J4':0.963, 'K4':0.903,
            #                           'B5':0.905, 'C5':0.897, 'D5':0.884, 'E5':0.996, 'F5':0.962, 'G5':0.897, 'H5':0.93, 'I5':0.907, 'J5':0.91, 'K5':0.91},
            #
            #      'aggregation_model': 'reward'  # 奖励比例聚合
            #  }


            # {'dataset': 'electricity', 'num_trainers': 100, 'split_type': 'no_action',
            #  'clients_num': 100,
            #  # 数据质量可看做各个参与方自主计算上传 计算可使用calculateDataQuality.py计算，下面的值即为计算结果
            #  'clients_data_quality': {'Bob': 0.949, 'Carol': 0.975, 'David': 0.945, 'Eve': 0.952, 'Frank': 0.945,
            #                           'Georgia': 0.886, 'Henry': 0.979, 'Isabel': 0.94, 'Joe': 0.788, 'kelly':0.963,
            #                           'B2': 0.942, 'C2': 0.951, 'D2': 0.994, 'E2': 0.546, 'F2': 0.946, 'G2': 0.944,'H2': 0.999, 'I2': 0.967, 'J2': 0.935, 'K2': 0.939,
            #                           'B3': 0.909, 'C3': 0.956, 'D3': 0.98, 'E3': 0.952, 'F3': 0.938, 'G3': 0.782,'H3': 0.933, 'I3': 0.897, 'J3': 0.952, 'K3': 0.984,
            #                           'B4': 0.947, 'C4': 0.948, 'D4': 0.871, 'E4': 0.96, 'F4': 0.951, 'G4': 0.936,'H4': 0.951, 'I4': 0.945, 'J4': 0.976, 'K4': 0.948,
            #                           'B5': 0.967, 'C5': 0.946, 'D5': 0.939, 'E5': 0.992, 'F5': 0.976, 'G5': 0.945, 'H5': 0.961, 'I5': 0.95, 'J5': 0.951, 'K5': 0.952,
            #                           'B6': 0.952, 'C6': 0.984, 'D6': 0.964, 'E6': 0.962, 'F6': 0.948, 'G6': 0.956, 'H6': 0.95, 'I6': 0.974, 'J6': 0.986, 'K6': 0.977,
            #                           'B7': 0.969, 'C7': 0.979, 'D7': 0.962, 'E7': 0.95, 'F7': 0.994, 'G7': 0.987, 'H7': 0.997, 'I7': 0.949, 'J7': 0.976, 'K7': 0.945,
            #                           'B8': 0.406, 'C8': 0.998, 'D8': 0.934, 'E8': 0.975, 'F8': 0.999, 'G8': 0.948, 'H8': 0.954, 'I8': 0.974, 'J8': 0.95, 'K8': 0.964,
            #                           'B9': 0.945, 'C9': 0.966, 'D9': 0.963, 'E9': 0.96, 'F9': 0.956, 'G9': 0.961, 'H9': 0.947, 'I9': 0.987, 'J9': 0.964, 'K9': 0.972,
            #                           'B10': 0.946, 'C10': 0.94, 'D10': 0.97, 'E10': 0.875, 'F10': 0.936, 'G10': 0.904, 'H10': 0.962, 'I10': 0.642, 'J10': 0.978, 'K10': 0.985
            #                           },
            #
            #  'aggregation_model': 'reward'  # 奖励比例聚合
            #  }

        ]

        method = 'step'  # 评估方法为step 
        seed = 89  # 随机种子
        for exp in experiments:  # 在不同拆分数据情况下进行实验
            print(f"Starting experiment with args: {exp}")
            runner.run_experiment(eval_method=method, seed=seed, **exp)
