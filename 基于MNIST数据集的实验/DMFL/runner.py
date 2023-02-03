# import joblib
import json
import threading
import time
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from DMFL.clients import TrainClient, IContractClient



class _Data:
    def __init__(self):
        self.data = []
        self.targets = []

    def split(self, n, ratios=None, flip_probs=None, disjointness=0):
        """
        使用给定的比率和标签翻转概率将数据集拆分为 n 个块。
        n：客户数量
        ratios：list拆分的大小比率 例[1,1,1,1]
        flip_probs：list标签翻转的比例（将data对应的target搞错）  例[0,0,0,0]  [0.1,0.3,0.2,0]
        disjointness：0-1 不相交比例； 0 = 随机拆分，1 = 按类不相交拆分  打乱（一个类别中混有其他类别数据）
        """
        if ratios is None:  # 输入的ratios flip_probs大小为n，如果没有输入，则默认ratios=[1]*n   flip_probs=[0]*n
            ratios = [1] * n
        if flip_probs is None:
            flip_probs = [0] * n
        if not n == len(ratios) == len(flip_probs):
            raise ValueError(f"Lengths of input arguments must match n={n}")

        # 按照类排序索引
        sorted_targets, sorted_idxs = torch.sort(self.targets)  # sort排序可参考https://blog.csdn.net/pearl8899/article/details/112184683
        perm = sorted_idxs.clone()  # clone()返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。

        print(f"\tsorted_targets={sorted_targets}")
        print(f"\t(counts)={torch.unique(sorted_targets, return_counts=True)}")  # torch.unique挑出tensor中的独立不重复元素  torch.unique用法https://blog.csdn.net/m0_45388819/article/details/121899915

        # 采用已排序的张量索引并打乱其中的一部分
        shuffle_proportion = 1 - disjointness  # 计算打乱比例
        shuffle_num = int(shuffle_proportion * len(sorted_idxs))  # 计算打乱张量索引的数目
        shuffle_idxs = torch.randperm(len(sorted_idxs))[:shuffle_num]  # 打乱索引下标 取前shuffle_num个  torch.randperm(n)将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
        sorted_shuffle_idxs, _ = torch.sort(shuffle_idxs)  # 将打乱的索引下标进行排序，得到sorted_shuffle_idxs
        for i, j in zip(sorted_shuffle_idxs, shuffle_idxs):  # perm中下标为i存的是打乱的索引值
            perm[i] = sorted_idxs[j]

        print(f"\tperm_targets={self.targets[perm]}")
        print(
            f"\t(counts)={torch.unique(self.targets[perm], return_counts=True)}")

        # 分成块
        num_chunks = sum(ratios)  # 计算总共的块数
        chunks = torch.chunk(perm, num_chunks)  # 用来将perm分成num_chunks个块 torch.chunk(tensor,chunk数,维数)

        chunk_it = 0
        data = []  # 每个client的数据存在这里
        targets = []  # 每个client的target存在这里
        for r, p in zip(ratios, flip_probs):  # 得到每个client的数据
            include_chunks = list(range(chunk_it, chunk_it+r))
            idxs = torch.cat([chunks[idxs] for idxs in include_chunks])  # 从按类排序好的总数据取出部分数据作为这个client的数据，待取出数据的索引即为idxs    torch.cat在给定维度上对输入的张量序列seq 进行连接操作
            chunk_it += r

            d = torch.index_select(input=self.data, dim=0, index=idxs)  # 根据索引取出属于该client的数据   torch.index_select根据index索引在input输入张量中选择某些特定的元素
            t = self._flip_targets(
                torch.index_select(
                    input=self.targets,
                    dim=0,
                    index=idxs
                ), p)  # 获取数据d对应的target（如果有打乱比例p，那么将按照比例来打乱data对应的target

            data.append(d)  # 向data中添加该client的数据
            targets.append(t)  #向targets中添加
        return data, targets

    def _flip_targets(self, targets, flip_p):  # 按照给定的打乱比例来打乱数据的target
        num_classes = len(torch.unique(targets))
        flip_num = int(flip_p * len(targets))
        flip_idx = torch.randperm(len(targets))[:flip_num]
        flipped_labels = torch.randint(
            low=0, high=num_classes, size=(len(flip_idx),), dtype=targets.dtype)  # 生成一个张量  形状为len(flip_idx)*1，数值为[0,num_classes)之间的随机整数
        for i, label in zip(flip_idx, flipped_labels):  # target中下标为i存的是随机的label值[0,num_classes)
            targets[i] = label
        return targets


class MNISTData(_Data):
    def __init__(self, train, subset=False, exclude_digits=None):
        self._dataset = datasets.MNIST(
            'experiments/mnist/resources',
            train=train)
        self.data = self._dataset.data.float().view(-1, 1, 28, 28) / 255
        self.targets = self._dataset.targets
        if subset:
            perm = torch.randperm(len(self.data))
            sub = perm[:(len(self.data)//200)]  # //表示整数除法，它可以返回商的整数部分（向下取整）
            self.data = self.data[perm]  # 这里perm是不是应该换为sub
            self.targets = self.targets[perm]  # 这里perm是不是应该换为sub
        if exclude_digits is not None:  # 排除一些数据
            for digit in exclude_digits:
                mask = self.targets != digit
                self.data = self.data[mask]
                self.targets = self.targets[mask]



class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ExperimentRunner:

    def __init__(
        self,
        quick_run,
        training_iterations,
        training_hyperparams,
        round_duration
    ):
        self.QUICK_RUN = quick_run  # 快速运行（是否取子集subset）
        self.TRAINING_ITERATIONS = training_iterations  # 训练迭代次数
        self.TRAINING_HYPERPARAMS = training_hyperparams  # 训练超参数
        self.ROUND_DURATION = round_duration  # 一轮的时间

    def run_experiment(
        self,
        dataset,
        split_type,  # 输入的参数
        eval_method,
        seed,

        clients_num,
        clients_data_quality,

        aggregation_model,

        num_trainers=3,
        ratios=None,
        flip_probs=None,
        disjointness=0,
        using_dp=None,
        unique_digits=None,
    ):
        # check args
        if split_type not in {'equal', 'size', 'flip', 'noniid', 'unique_digits', 'dp', 'size+noniid'}:
            raise KeyError(f"split_type={split_type} is not a valid option")

        # 制作结果字典，添加当前实验的详细信息
        results = {}
        results['dataset'] = dataset
        results['quick_run'] = self.QUICK_RUN
        results.update(self.TRAINING_HYPERPARAMS)  # 将self.TRAINING_HYPERPARAMS添加到results  字典dict1.update(dict2)将字典dict2内容更新到dict1中
        results['split_type'] = split_type
        results['seed'] = seed
        results['num_trainers'] = num_trainers
        if ratios is not None:
            results['ratios'] = ratios
        if flip_probs is not None:
            results['flip_probs'] = flip_probs
        results['disjointness'] = disjointness
        if using_dp is None:
            using_dp = [False] * num_trainers
        results['using_dp'] = using_dp
        if unique_digits is not None:
            results['unique_digits'] = unique_digits

        # set up  设置
        torch.manual_seed(seed)  # 设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象  这样每次重新运行程序时，torch.rand生成的随机数跟上一次运行程序时一样
        alice, trainers = self._make_clients(
            dataset, split_type, num_trainers, ratios, flip_probs, disjointness, unique_digits, clients_num, clients_data_quality, aggregation_model)  # 产生clients
        results['contract_address'] = alice.contract_address  #
        results['trainers'] = [trainer.name for trainer in trainers]


        # 每个参与方trainer上传msg.sender到智能合约
        for index, trainer in enumerate(trainers):
            trainer.uploadMsgSender(index+1)  #调用clients中的uploadMsgSender，tx在clients中已处理




        if dataset == 'mnist':
            results['digit_counts'] = self._label_counts(10, trainers)

        # define training threads  定义训练线程
        threads = []
        for trainer, dp in zip(trainers, using_dp):
            train_hyperparams = self.TRAINING_HYPERPARAMS
            threads.append(
                threading.Thread(  # threading.Thread  target是被 run()方法调用的回调对象  kwargs传递参数，目标调用的参数的关键字dictionary，默认为{}
                    target=trainer.train_until,
                    kwargs=train_hyperparams,
                    daemon=True
                )
            )


        #  定义评估线程
        threads.append(
            threading.Thread(
                target=alice.evaluate_until,
                args=(self.TRAINING_ITERATIONS, eval_method),
                daemon=True
            )
        )

        #  并行运行所有线程
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        #  得到结果并记录下来
        results['global_loss'] = [alice.evaluate_global(i)  # 记录每轮全局模型的损失
                                  for i in range(1, self.TRAINING_ITERATIONS+2)]
        one_hot_output = (dataset == 'mnist')  # 如果数据集为mnist one_hot_output=1 否则one_hot_output=0
        results['final_global_accuracy'] = self._global_accuracy(  # 计算当前全局模型的准确度   _global_accuracy中用到了client.predict 这个应该在consortium协议中才能使用
            alice, one_hot_output)

        results['token_counts'] = self._token_count_histories(trainers)  # 返回每个trainer在每轮中获取的token数目
        results['total_token_counts'] = [alice.get_total_token_count(i)
                                         for i in range(1, self.TRAINING_ITERATIONS+1)]  # 返回每轮所有trainer在指定轮所获得的所有token数目
        results['gas_used'] = self._gas_used(alice, trainers)  # 返回评估者alice和每个trainer整个过程消耗的gas
        results['gas_history'] = self._gas_history(alice, trainers)

        self._save_results(results)  # 保存结果

        return results

    def _make_clients(
        self,
        dataset,
        split_type,
        num_trainers,
        ratios,
        flip_probs,
        disjointness,
        unique_digits,

        clients_num,
        clients_data_quality,

        aggregation_model
    ):
        #  选择数据集及其对应模型
        if dataset == 'mnist': 
            dataset_constructor = MNISTData
            model_constructor = MNISTModel
            model_criterion = F.nll_loss
        else:
            raise KeyError(f"Invalid dataset key: {dataset}")
        alice_dataset = dataset_constructor(train=False, subset=self.QUICK_RUN)  # 这里获取的是数据集中的测试数据 train=False
        alice_data = alice_dataset.data
        alice_targets = alice_dataset.targets

        alice = IContractClient(  ##
            name="Alice",
            data=alice_data,
            targets=alice_targets,
            model_constructor=model_constructor,
            model_criterion=model_criterion,

            clients_num=clients_num,
            clients_data_quality=clients_data_quality,

            aggregation_model = aggregation_model,

            account_idx=0,
            contract_address=None,
            deploy=True,
        )
        trainers = self._make_trainers(
            dataset_constructor=dataset_constructor,
            model_constructor=model_constructor,
            model_criterion=model_criterion,
            split_type=split_type,
            num_trainers=num_trainers,
            clients_data_quality=clients_data_quality,
            aggregation_model = aggregation_model,
            ratios=ratios,
            flip_probs=flip_probs,
            disjointness=disjointness,
            unique_digits=unique_digits,
            client_constructor=TrainClient, ##
            contract_address=alice.contract_address)
        alice.set_genesis_model(
            round_duration=self.ROUND_DURATION,
            max_num_updates=len(trainers)
        )
        return alice, trainers

    def _make_trainers(
        self,
        dataset_constructor,
        model_constructor,
        model_criterion,
        split_type,
        num_trainers,
        clients_data_quality,
        aggregation_model,
        ratios,
        flip_probs,
        disjointness,
        unique_digits,
        client_constructor,
        contract_address
    ):
        # instantiate data  实例化数据  得到data target
        if split_type == 'equal' or split_type == 'dp':  # 将数据平均分为num_trainers部分
            data, targets = dataset_constructor(
                train=True, subset=self.QUICK_RUN).split(num_trainers)
        if split_type == 'size':  # 将数据分为num_trainers部分，按照给定的拆分大小比例ratios
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, ratios=ratios)
        if split_type == 'flip':  # 将数据平均分为num_trainers部分，按照flip_probs打乱target（产生错误target数据）
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, flip_probs=flip_probs)
        if split_type == 'noniid':  # 将数据平均分为num_trainers部分，按照disjointness（不相交比例，取值范围[0,1]）对数据进行打乱（其他类别数据会混进某一类别中）
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, disjointness=disjointness)


        if split_type == 'size+noniid':  # non-iid 且 每部分的数据数量不同
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, ratios=ratios, disjointness=disjointness)


        if split_type == 'unique_digits':  # 只针对MNISTData数据集   
            assert dataset_constructor == MNISTData, "split_type=unique_digits only supported for MNISTData"
            dataset_unique = dataset_constructor(train=True, subset=self.QUICK_RUN,
                                                 exclude_digits=set(range(10))-set(unique_digits))  # 得到unique数据
            data_unique = dataset_unique.data
            targets_unique = dataset_unique.targets
            data_others, targets_others = MNISTData(
                train=True, subset=self.QUICK_RUN, exclude_digits=unique_digits
            ).split(num_trainers-1) # 对除去unique的数据进行拆分（平均分为num_trainers-1个部分）
            data = [data_unique] + data_others  # 分好的数据data再加上[data_unique]
            targets = [targets_unique] + targets_others  # 分好的数据target再加上[target_unique]

        clients_data_quality_list = []  # 各个参与方数据质量的值组成的列表
        for q in clients_data_quality.values():
            clients_data_quality_list.append(q)
        index = np.argsort(clients_data_quality_list)
        IContractTypeOrder = np.argsort(index) # 各个参与方应该选择的合同类型，从0开始的
        print(f"各个参与方应该选择的合同类型列表：{IContractTypeOrder}")

        # instantiate clients  实例化client
        trainers = []  # trainers
        names = ["Bob", "Carol", "David", "Eve",
                 "Frank", "Georgia", "Henry", "Isabel", "Joe", "Kelly"]
        for i, name, d, t in zip(range(num_trainers), names, data, targets):
            trainer = client_constructor(  # 结合clients.py实例化client
                name=name,
                data=d,
                targets=t,
                account_idx=i+1,
                model_constructor=model_constructor,
                model_criterion=model_criterion,
                contract_address=contract_address,

                icType = int(IContractTypeOrder[i]+1),   # 注意这样设定合同类型前提是参与方是按照数据质量从小到大实例化的
                                # 为每个参与方设定选择的合同类型 后期可以传入数据质量，让参与方根据数据质量进行自主选择
                clients_num = num_trainers,  # 参与方的数量

                aggregation_model = aggregation_model,

                deploy=False
            )
            print(f"\t{name} counts: {torch.unique(t, return_counts=True)}")  # client的名称 该client有多少不同的target
            trainers.append(trainer)  # 向trainers中添加该client即trainer
        return trainers

    def _global_accuracy(self, client, one_hot_output):  # 计算当前全局模型的准确度
        output = client.predict()
        if one_hot_output:
            pred = output.argmax(dim=2, keepdim=True).squeeze()
        else:
            pred = torch.round(output).squeeze()
        num_correct = (pred == client._targets).float().sum().item()
        accuracy = num_correct / len(pred)
        return accuracy

    def _label_counts(self, num_classes, trainers):  # 返回每个trainer数据中的不同类别对应的数目
        digit_counts_by_name = {}
        for trainer in trainers:
            digit_counts = [0]*num_classes
            digits, counts = torch.unique(
                trainer._targets, return_counts=True)  # hacky   torch.unique 参考https://blog.csdn.net/m0_45388819/article/details/121899915   
                                                       # 返回第一个张量为排序后的数据不重复元素，第二个张量为每个元素出现的次数
            digits = digits.int()
            print("**********************")
            print(digits)
            for digit, count in zip(digits, counts):
                digit_counts[digit] = count.item()
            digit_counts_by_name[trainer.name] = digit_counts
        return digit_counts_by_name

    def _token_count_histories(self, trainers):  # 返回每个trainer在每轮中获取的token数目
        token_count_history_by_name = {}
        for trainer in trainers:
            token_count_history_by_name[trainer.name] = [
                trainer.get_token_count(training_round=i)
                for i in range(1, self.TRAINING_ITERATIONS+1)
            ]
        return token_count_history_by_name

    def _gas_used(self, alice, trainers):  # 返回评估者alice和每个trainer整个过程（所有轮结束）消耗的gas
        gas_used_by_name = {}
        gas_used_by_name[alice.name] = alice.get_gas_used()
        for trainer in trainers:
            gas_used_by_name[trainer.name] = trainer.get_gas_used()
        return gas_used_by_name

    def _gas_history(self, alice, trainers):  # 返回评估者alice和每个trainer每轮消耗的gas  alice第二轮消耗gas才开始对应trainer的第一轮训练
        gas_history_by_name = {}
        gas_history_by_name[alice.name] = alice.get_gas_history()
        for trainer in trainers:
            gas_history_by_name[trainer.name] = trainer.get_gas_history()
        return gas_history_by_name

    def _save_results(self, results):  # 将训练信息保存到文件中
        dataset = results['dataset']
        filedir = f"experiments/{dataset}/results/"
        if self.QUICK_RUN:  # 如果是快速运行的实验，那么结果存储在quick文件夹下
            filedir += "quick/"
        filename = "all.json"
        filepath = filedir + filename
        with open(filepath) as f:
            all_results = json.load(f)  # 读取文件内容，读取的结果返回为python的dict对象
        all_results.append(results)
        with open(filepath, 'w') as f:  # 写模式
            json.dump(all_results, f,  # json.dump将python对象转换为适当的json对象
                      indent=4,
                      sort_keys=True)
        print(f"{filepath} now has {len(all_results)} results")



