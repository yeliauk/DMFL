import threading
import time
import methodtools
import syft as sy
import torch

from DMFL.ipfs_client import IPFSClient
from DMFL.contract_clients import ContractClient

import numpy as np

_hook = sy.TorchHook(torch)  # pysyft需要hook

begin_evaluate = 0  # 开始评估类的工作
begin_train = 0  #开始训练类的工作


class _BaseClient:
    """
    包含客户端的共同特征的抽象基础客户端
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None, deploy=False):
        self.name = name
        self._model_constructor = model_constructor
        if deploy:
            self._print("Deploying contract...")
        self._contract = contract_constructor(
            account_idx, contract_address, deploy)
        self._account_idx = account_idx
        self.address = self._contract.address  # 地址(这个实例化合约对象的地址，不知道对不对，先这样理解，每实例化一个合约，那么这个地址就不一样)
        self.contract_address = self._contract.contract_address  # 合约地址  （合约地址，在该代码中合约创建者Alice和trainers的合约地址应该是一样的）
        self._print(
            f"Connected to contract at address {self.contract_address}")

    def get_token_count(self, address=None, training_round=None):  # 获取某个trainer指定轮所获得的所有token数目
        return self._contract.countTokens(address, training_round)

    def get_total_token_count(self, training_round=None):  # 获取所有trainer指定轮所获得的所有token数目
        return self._contract.countTotalTokens(training_round)



    def get_current_token_count(self, address=None, training_round=None):  # 获取某个trainer指定轮这一轮所获得的的奖励数目
        return self._contract.countCurrentToken(address, training_round)

    def get_current_total_count(self, training_round=None):  # 获取所有trainer指定轮这一轮获得的所有奖励数目
        return self._contract.countCurrentTotalTokens(training_round)



    def get_gas_used(self):
        return self._contract.get_gas_used()

    def wait_for_txs(self, txs):
        receipts = []
        if txs:
            self._print(f"Waiting for {len(txs)} transactions...")
            for tx in txs:
                receipts.append(self._contract.wait_for_tx(tx))
            self._print(f"{len(txs)} transactions mined")
        return receipts

    # 上传trainer的账户地址，保存到智能合约   ***
    def uploadMsgSender(self, index):
        tx = self._contract.trainerMsgSenderRecord(index)
        self.wait_for_txs([tx])



    def _print(self, msg):
        print(f"{self.name}: {msg}")


class _GenesisClient(_BaseClient):
    """
    扩展基础客户端，能够设置创世模型以开始训练。
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None, deploy=False):
        super().__init__(name, model_constructor,
                         contract_constructor, account_idx, contract_address, deploy)
        self._ipfs_client = IPFSClient(model_constructor)

    def set_genesis_model(self, round_duration, max_num_updates=0):
        self._print("Setting genesis...")
        genesis_model = self._model_constructor().cuda()  # genesis模型  初始模型（初始全局模型，由alice创建）  !!!!
        genesis_cid = self._upload_model(genesis_model)  # 上传genesis模型参数到IPFS，并返回对应Hash值
        tx = self._contract.setGenesis(
            genesis_cid, round_duration, max_num_updates)
        self.wait_for_txs([tx])

    def _upload_model(self, model):  # 将模型上传到IPFS，并返回对应的Hash值
        """将给定模型上传到 IPFS。"""
        uploaded_cid = self._ipfs_client.add_model(model)
        return uploaded_cid


class TrainClient(_GenesisClient):
    """
    参与联邦学习训练的客户端（即参与方）
    """

    CURRENT_ROUND_POLL_INTERVAL = 1.  # Ganache can't mine quicker than once per second   Ganache的挖矿速度不能超过每秒一次

    def __init__(self, name, data, targets, model_constructor, model_criterion, icType, clients_num, aggregation_model, account_idx, contract_address=None, deploy=False):
        super().__init__(name,
                         model_constructor,
                         ContractClient,
                         account_idx,
                         contract_address,
                         deploy)
        self.data_length = min(len(data), len(targets))

        self._worker = sy.VirtualWorker(_hook, id=name)
        self._criterion = model_criterion
        self._data = data
        self._targets = targets
        # self._test_loader = torch.utils.data.DataLoader(
        #     sy.BaseDataset(self._data, self._targets),
        #     batch_size=len(data)
        # )  # 测试数据
        # train loader is defined each time training is run 训练装载程序在每次训练时定义
        
        self._gas_history = {}

        self.icType = icType  # 该参与方选择的合同类型
        self.clients_num = clients_num

        self.aggregation_model = aggregation_model # 聚合方式

    def train_until(self, final_round_num, batch_size, epochs, learning_rate):  # 进行多轮训练（有给定的最大训练轮数）
        global begin_train
        global begin_evaluate
        start_round = self._contract.currentRound()
        for r in range(start_round, final_round_num+1):
            self.wait_for_round(r)

            while True:
                if begin_train > 0:
                    begin_train = begin_train - 1
                    break

            txs1 = self.signIcontract(self.icType, r)  # 签订合同（已指定合同类型）
            self.wait_for_txs(txs1)

            tx = self._train_single_round(  # 每一轮的训练
                r,
                batch_size,
                epochs,
                learning_rate
            )
            self.wait_for_txs([tx])
            self._gas_history[r] = self.get_gas_used()

            begin_evaluate = begin_evaluate+1

        self._print(f"Done training. Gas used: {self.get_gas_used()}")


    def get_current_global_model(self):  # 获取当前训练轮的全局模型
        """
        计算或从缓存中获取当前全局模型。
        """
        current_training_round = self._contract.currentRound()
        current_global_model = self._get_global_model(current_training_round)
        return current_global_model

    def wait_for_round(self, n):  # 等待n轮的开始
        self._print(
            f"Waiting for round {n} ({self._contract.secondsRemaining()} seconds remaining)...")
        while(self._contract.currentRound() < n):
            time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)
        self._print(f"Round {n} started")

    def get_gas_history(self):
        return self._gas_history

    @methodtools.lru_cache()
    def _get_global_model(self, training_round):  #  根据给定训练轮数计算并返回全局模型（注意根据上一轮的更新来计算给定轮的全局模型）
        if self.aggregation_model == 'avg': # fedavg聚合
            """
            通过聚合上一轮的更新来计算给定训练轮的全局模型。
            仅当 training_round 与当前合同培训轮次匹配时才能执行此操作。
            """
            model_cids = self._get_cids(training_round - 1)  # 注意这个细节：根据上一轮的更新
            models = self._get_models(model_cids)
            avg_model = self._avg_model(models)
            return avg_model
        if self.aggregation_model == 'reward':
            """
            通过聚合上一轮的更新来计算给定训练轮的全局模型,聚合比例按照各个参与方所获奖励占总奖励的比例
            """
            reward_avg_model = self._reward_avg_model(training_round, 1)  # 1：按照指定轮这一轮的奖励比例进行聚合   0：按照指定轮以及之前轮获得的所有奖励比例
            return reward_avg_model

    def _train_single_round(self, round_num, batch_size, epochs, learning_rate):  # 进行一轮训练（视角为参与方）
        """
        使用本地数据进行一轮训练，上传并记录贡献。
        """
        model = self.get_current_global_model()  # 获取当前轮的全局模型（包括获取上一轮的参与方模型，进行全局模型的聚合）
        self._print(f"Training model, round {round_num}...")
        model = self._train_elec_model(
            model, batch_size, epochs, learning_rate)  # 开始本地训练  用电量数据
        uploaded_cid = self._upload_model(model)  # 上传本地训练模型到IPFS，得到对应的Hash值
        self._print(f"Adding model update...")
        tx = self._record_model(uploaded_cid, round_num)  # 将本轮Hash值上传到智能合约，进行记录
        return tx

    def _train_model(self, model, batch_size, epochs, lr):  # 训练模型（为本地训练）
        train_loader = torch.utils.data.DataLoader(
            sy.BaseDataset(self._data, self._targets),
            batch_size=batch_size)  # 加载训练数据
        # model = model.send(self._worker)
        model.train()
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr
        )
        # 模型训练************************************
        for epoch in range(epochs):
            for data, labels in train_loader:
                optimizer.zero_grad()
                pred = model(data)
                loss = self._criterion(pred, labels)
                loss.backward()
                optimizer.step()
        # model.get()
        return model

    def _train_elec_model(self, model, batch_size, epochs, lr):  # 训练用电量模型（为本地训练）
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
        torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, last_epoch=-1)

        # 模型训练************************************
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            item_count = 0
            for b in range(0, len(self._data), batch_size):
                inpt = self._data[b:b + batch_size, :, :]
                target = self._targets[b:b + batch_size]

                x_batch = torch.tensor(inpt, dtype=torch.float32).cuda()
                y_batch = torch.tensor(target, dtype=torch.float32).cuda()

                model.init_hidden(x_batch.size(0))
                #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
                #    lstm_out.contiguous().view(x_batch.size(0),-1)
                output = model(x_batch)
                loss = self._criterion(output.view(-1), y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item() * len(y_batch)
                item_count += len(y_batch)
        return model

    def _record_model(self, uploaded_cid, training_round):  #在智能合约上记录指定轮的各个参与方模型参数对应的Hash值
        """在智能合约上记录给定模型 IPFS Hash值。"""
        return self._contract.addModelUpdate(uploaded_cid, training_round)

    def _get_cids(self, training_round):  # 返回指定轮的参与方模型参数Hash值
        if training_round < 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 0:
            return [self._contract.genesis()]  # 返回genesis模型的hash
        cids = self._contract.updates(training_round)  # 返回指定训练轮的各个参与方模型参数hash [*** *** ... ***]
        if not cids:  # 如果 cids 为空，请参考上一轮
            return self._get_cids(training_round - 1)  #重新调用该函数来返回上一轮的参与方模型参数Hash
        return cids

    def _get_models(self, model_cids):  # 根据若干个Hash值返回对应的模型
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid)
            models.append(model)
        return models

    def _avg_model(self, models):  # 传入若干个模型，计算全局聚合模型（采用平均算法聚合各个模型参数），返回全局聚合模型
        avg_model = self._model_constructor().cuda()  #!!!!
        with torch.no_grad():  # 可以显著降低显存
            for params in avg_model.parameters():  # model.parameters()保存的是Weights和Bais参数的值
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    def _reward_avg_model(self, training_round, isCurrentRoundReward):  # 传入若干个模型，计算全局聚合模型（采用各个参与方获得奖励的比例聚合各个模型参数），返回全局聚合模型
        if training_round <= 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 1:  #  这是第一轮训练，使用初始化的模型
            modelsHash = self._get_cids(0)
            round1_models = self._get_models(modelsHash)
            avg_model = self._avg_model(round1_models)
            return avg_model
        # modelsHash_ratio = {}  # 记录各个参与方模型hash与其对应的聚合比例
        models = []  # 存放各个参与方该轮训练的本地模型
        ratios = []  # 存放计算全局模型聚合时各个参与方模型参数的所占比例
        if isCurrentRoundReward == 1:  # 按照给定轮这一轮的奖励计算比例
            # 获取所有参与方在给定轮获得的奖励数目
            trainers_current_reward = self.get_current_total_count(training_round - 1)
        else:  # 按照给定轮以及之前所获得的的所有奖励
            # 获取所有参与方指定轮所获得的所有奖励数目
            trainers_reward = self.get_total_token_count(training_round - 1)
        for i in range(1, self.clients_num+1):
            # 获取参与方msg.sender
            address = self._contract.getTrainerMsgSender(i)
            # 获取给定轮和指定trainer的模型hash
            modelHash = self._contract.getTrainerHash(address, training_round-1)
            # 获取模型
            model_list = self._get_models(modelHash)
            models.append(model_list[0])
            if isCurrentRoundReward == 1:
                # 获取所有参与方在给定轮获得的奖励数目
                trainer_current_reward = self.get_current_token_count(address, training_round-1)
                # 计算全局模型聚合时该参与方模型参数的所占比例
                ratio = trainer_current_reward / trainers_current_reward
            else:
                # 获取某个参与方指定轮所获得的所有奖励数目
                trainer_reward = self.get_token_count(address, training_round-1)
                # 计算全局模型聚合时该参与方模型参数的所占比例
                ratio = trainer_reward / trainers_reward
            ratios.append(ratio)

        print(f"第{training_round}轮 全局模型聚合时各个参与方模型参数应占比例：{ratios}")

        reward_avg_model = self._model_constructor().cuda()  # !!!!
        with torch.no_grad():  # 可以显著降低显存
            for params in reward_avg_model.parameters():
                params *= 0
            for (client_model, r) in zip(models, ratios):
                for avg_param, client_param in zip(reward_avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param * r
        return reward_avg_model


    # 签订合同（已选定类型）
    def signIcontract(self, icType, round):
        txs = []
        # 从智能合约获取选定类型合同的注册费
        fee = self._contract.getAFee(icType, round)
        # 签订选择的合同类型，并缴纳费用
        tx = self._contract.signContract(icType, fee)
        txs.append(tx)
        return txs


class incentiveContract:
    """
    基于合同理论的合同类
    """
    def __init__(self, clients_num, clients_data_quality):  #参与方数目，参与方的数据质量（参与方名称与其数据质量）
        self.clients_num = clients_num
        self.clients_data_quality = clients_data_quality  # 字典{'a': 0.78, 'b': 0.8, ...}
        self.sort_clients_data_quality = self.sort_data_quality(clients_data_quality)

    def sort_data_quality(self, clients_data_quality):  # 将参与方数据质量按照从小到大排序（按照value进行排序），返回list类型
        return sorted(clients_data_quality.items(), key=lambda x: x[1])  # 列表 [('a', 0.78), ('b', 0.8), ...]


    def product_incentiveContracts(self, round):  #根据训练轮数，根据公式计算产生为各个参与方设定的合同
        c = 50
        theta = []  # 各个参与方的数据质量（排序后的 从小到大）
        for client_data_quality in self.sort_clients_data_quality:
            theta.append(client_data_quality[1])
        print(f"theta:{theta}")
        M = [i / 20 for i in theta]  # 准确率标准 list
        print(f"M:{M}")
        GM = [i * i * 4000 for i in M]  # 给整体带来的贡献
        print(f"GM:{GM}")
        R = GM  # 每个合同中的奖励值

        f = [0] * self.clients_num  # 每个合同中的注册费
        f[0] = (1 / (2 * c)) * pow((theta[0] * R[0]), 2)
        for i in range(1, self.clients_num):
            f[i] = (1 / (2 * c)) * pow((theta[i] * R[i]), 2) - (1 / (2 * c)) * pow((theta[i] * R[i - 1]), 2) + f[i - 1]
        print(f"R:{R}")
        print(f"f:{f}")

        M = [int(i * 10000) for i in M]  # 为了方便存储扩大了10000倍
        R = [int(i * 10000) for i in R]  # 为了方便存储扩大了10000倍
        f = [int(i * 100000) for i in f]  # 为了方便存储扩大了10000倍
        print("扩大为整数后：")
        print(f"M:{M}")
        print(f"R:{R}")
        print(f"f:{f}")

        iContracts = {i + 1: (f[i], R[i]) for i in range(self.clients_num)}  # 组合为固定的合同格式 {1:(f1,R1), 2:(f2,R2), ...}
        iModelLine = {i + 1: M[i] for i in range(self.clients_num)}  # 各个参与方本地模型准确率标准线
        print(f"iContracts:{iContracts}")
        print(f"iModelLine:{iModelLine}")
        return iContracts, iModelLine



class IContractClient(_GenesisClient):
    """
    对各个参与方的表现进行评估（即合同发布者）
    """
    TOKENS_PER_UNIT_LOSS = 1e18  # same as the number of wei per ether   每个以太的wei数相同 1*10^18
    # 每个减少的单位loss对应的奖励
    CURRENT_ROUND_POLL_INTERVAL = 1.  # Ganache can't mine quicker than once per second   Ganache的挖矿速度不能超过每秒一次

    def __init__(self, name, data, targets, model_constructor, model_criterion,
                 clients_num, clients_data_quality, aggregation_model,
                 account_idx, contract_address=None,
                 deploy=False):
        super().__init__(name,
                         model_constructor,
                         ContractClient,
                         account_idx,
                         contract_address,
                         deploy)
        self.data_length = min(len(data), len(targets))

        self._worker = sy.VirtualWorker(_hook, id=name)
        self._criterion = model_criterion
        self._data = data  # .send(self._worker)
        self._targets = targets  # .send(self._worker)
        self._test_loader = torch.utils.data.DataLoader(
            sy.BaseDataset(self._data, self._targets),
            batch_size=len(data)   # !!!!
        )  # 测试数据
        # train loader is defined each time training is run 训练装载程序在每次训练时定义

        self._gas_history = {}

        self.clients_num = clients_num
        self.aggregation_model = aggregation_model
        self.incentiveContract = incentiveContract(clients_num, clients_data_quality)  # 合同类实例化


    def evaluate_until(self, final_round_num, method):  # 对所有训练轮数进行评估，分配奖励  线程间交互变量待考虑
        global begin_evaluate
        global begin_train
        clients_num = self.clients_num
        begin_evaluate = clients_num
        while True:
            if begin_evaluate == clients_num:
                begin_evaluate = 0
                break

        print("准备发布合同")
        txs = self._product_icontract(1)  #发布第1轮合同
        print("发布合同完成")
        self.wait_for_txs(txs)
        self._gas_history[1] = self.get_gas_used()
        print("事务记录处理完毕")

        begin_train = clients_num

        for r in range(1, final_round_num + 1):
            self.wait_for_round(r + 1)

            while True:
                if begin_evaluate == clients_num:
                    begin_evaluate = 0
                    break

            hash_reward = self._evaluate_one_round(r)
            txs1 = self._set_reward(hash_reward)
            self.wait_for_txs(txs1)
            if r+1 <= final_round_num:
                txs2 = self._product_icontract(r + 1)  # 发布新一轮的合同
                self.wait_for_txs(txs2)
            self._gas_history[r + 1] = self.get_gas_used()

            begin_train = clients_num

        self._print(f"Done evaluating. Gas used: {self.get_gas_used()}")


    def get_current_global_model(self):  # 获取当前训练轮的全局模型  runner中predict 计算当前全局模型准确度时用到了
        """
        计算或从缓存中获取当前全局模型。
        """
        current_training_round = self._contract.currentRound()
        current_global_model = self._get_global_model(current_training_round)
        return current_global_model

    @methodtools.lru_cache()
    def evaluate_global(self, training_round):  # 评估指定轮的全局模型，返回loss
        model = self._get_global_model(training_round)
        deviation = self.elec_test(model)  # 用电量数据 模型误差计算
        return deviation

    def predict(self):  # 使用当前全局模型预测测试集，并返回预测结果
        model = self.get_current_global_model()
        # model = model.send(self._worker)
        predictions = []
        with torch.no_grad():
            for data, labels in self._test_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)  # .get()
                predictions.append(pred)
        return torch.stack(predictions)


    def wait_for_round(self, n):  # 等待n轮的开始
        self._print(
            f"Waiting for round {n} ({self._contract.secondsRemaining()} seconds remaining)...")
        while (self._contract.currentRound() < n):
            time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)
        self._print(f"Round {n} started")

    def get_gas_history(self):
        return self._gas_history

    def _reward_avg_model(self, training_round, isCurrentRoundReward):  # 传入若干个模型，计算全局聚合模型（采用各个参与方获得奖励的比例聚合各个模型参数），返回全局聚合模型
        if training_round <= 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 1:  # 这是第一轮训练，使用初始化的模型
            modelsHash = self._get_cids(0)
            round1_models = self._get_models(modelsHash)
            avg_model = self._avg_model(round1_models)
            return avg_model
        # modelsHash_ratio = {}  # 记录各个参与方模型hash与其对应的聚合比例
        models = []  # 存放各个参与方该轮训练的本地模型
        ratios = []  # 存放计算全局模型聚合时各个参与方模型参数的所占比例
        if isCurrentRoundReward == 1:  # 按照给定轮这一轮的奖励计算比例
            # 获取所有参与方在给定轮获得的奖励数目
            trainers_current_reward = self.get_current_total_count(training_round - 1)
        else:  # 按照给定轮以及之前所获得的的所有奖励
            # 获取所有参与方指定轮所获得的所有奖励数目
            trainers_reward = self.get_total_token_count(training_round - 1)
        for i in range(1, self.clients_num + 1):
            # 获取参与方msg.sender
            address = self._contract.getTrainerMsgSender(i)
            # 获取给定轮和指定trainer的模型hash
            modelHash = self._contract.getTrainerHash(address, training_round - 1)
            # 获取模型
            model_list = self._get_models(modelHash)
            models.append(model_list[0])
            if isCurrentRoundReward == 1:
                # 获取所有参与方在给定轮获得的奖励数目
                trainer_current_reward = self.get_current_token_count(address, training_round - 1)
                # 计算全局模型聚合时该参与方模型参数的所占比例
                ratio = trainer_current_reward / trainers_current_reward
            else:
                # 获取某个参与方指定轮所获得的所有奖励数目
                trainer_reward = self.get_token_count(address, training_round - 1)
                # 计算全局模型聚合时该参与方模型参数的所占比例
                ratio = trainer_reward / trainers_reward
            ratios.append(ratio)

        print(f"*第{training_round}轮 全局模型聚合时各个参与方模型参数应占比例：{ratios}")

        reward_avg_model = self._model_constructor().cuda()  # !!!!
        with torch.no_grad():  # 可以显著降低显存
            for params in reward_avg_model.parameters():
                params *= 0
            for (client_model, r) in zip(models, ratios):
                for avg_param, client_param in zip(reward_avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param * r
        return reward_avg_model

    @methodtools.lru_cache()  # 简单的理解为保存多次执行结果，当传入某一参数的执行结果已经执行过，则不会再一次执行而是直接返回结果
    def _get_global_model(self, training_round):  # 根据给定训练轮数计算并返回全局模型（注意根据上一轮的更新来计算给定轮的全局模型）
        if self.aggregation_model == 'avg':
            """
            通过聚合上一轮的更新来计算给定训练轮的全局模型。
            仅当 training_round 与当前合同培训轮次匹配时才能执行此操作。
            """
            model_cids = self._get_cids(training_round - 1)  # 注意这个细节：根据上一轮的更新
            models = self._get_models(model_cids)
            avg_model = self._avg_model(models)
            return avg_model
        if self.aggregation_model == 'reward':
            """
            通过聚合上一轮的更新来计算给定训练轮的全局模型,聚合比例按照各个参与方所获奖励占总奖励的比例
            """
            reward_avg_model = self._reward_avg_model(training_round, 1)
            return reward_avg_model

    def _evaluate_model(self, model):  # 使用测试数据评估给定的模型，返回loss值（平均）
        model = model  # .send(self._worker)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data, labels in self._test_loader:
                pred = model(data)  # 使用该模型对测试数据进行预测
                total_loss += self._criterion(pred, labels
                                              ).item()
                # ).get().item()
        avg_loss = total_loss / len(self._test_loader)
        return avg_loss

    def elec_test(self, model):  # 使用测试数据评估给定的用电量预测模型
        n_timesteps = 30
        n_features = 1
        model.eval()
        model.init_hidden(1)  # 因为是测试，所以每次只输入1个样本

        pred_y_list = list()
        for x in self._data:
            input = torch.from_numpy(x)
            x_batch = torch.tensor(input, dtype=torch.float32).view(1, n_timesteps, n_features).cuda()
            pred_y_list.append(model(x_batch).data.cpu().numpy())

        print(pred_y_list[:5])
        print(self._targets[:5])

        # MAE
        temp_list1 = [np.abs(pred_y_list[i] - self._targets[i]) for i in range(len(pred_y_list))]
        mae = (np.sum(temp_list1) / len(pred_y_list))

        # RMSE
        temp_list2 = [(pred_y_list[i] - self._targets[i]) ** 2 for i in range(len(pred_y_list))]
        rmse_test = (np.sum(temp_list2) / len(pred_y_list)) ** 0.5

        # MSE
        mse_test = rmse_test ** 2

        # R方
        R_square = 1 - mse_test / np.var(self._targets)

        return rmse_test


    def _get_cids(self, training_round):  # 返回指定轮的参与方模型参数Hash值
        if training_round < 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 0:
            return [self._contract.genesis()]  # 返回genesis模型的hash
        cids = self._contract.updates(training_round)  # 返回指定训练轮的各个参与方模型参数hash [*** *** ... ***]
        if not cids:  # if cids is empty, refer to previous round  如果 cids 为空，请参考上一轮
            return self._get_cids(training_round - 1)  # 重新调用该函数来返回上一轮的参与方模型参数Hash
        return cids

    def _get_models(self, model_cids):  # 根据若干个Hash值返回对应的模型
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid)
            models.append(model)
        return models

    def _avg_model(self, models):  # 传入若干个模型，计算全局聚合模型（采用平均算法聚合各个模型参数），返回全局聚合模型
        avg_model = self._model_constructor().cuda()  # !!!!
        with torch.no_grad():  # 可以显著降低显存
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model


    def _set_reward(self, hash_reward):  # 根据某一轮参与方的分数给他们分配token

        txs = []
        self._print(f"Setting {len(hash_reward.values())} reward...")
        for hash, reward in hash_reward.items():
            tx = self._contract.setTokens(hash, reward)  # 分配token
            txs.append(tx)
        return txs



    def _get_trainer_model_accuracy(self, model, one_hot_output=1):  # 测试某个参与方本地模型的准确率   one_hot_output 0/1  1代表数据集为mnist
        predictions = []
        with torch.no_grad():
            for data, labels in self._test_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)  # .get()
                predictions.append(pred)
        output = torch.stack(predictions)
        if one_hot_output:
            pred = output.argmax(dim=2, keepdim=True).squeeze()
        else:
            pred = torch.round(output).squeeze()
        num_correct = (pred == self._targets).float().sum().item()
        accuracy = num_correct / len(pred)
        return accuracy  # 小数 0.***


    def _product_icontract(self, round):  # 发布该轮合同，并将合同内容上传到 智能合约
        iContracts, iModelLine = self.incentiveContract.product_incentiveContracts(round)  # 根据轮数产生合同 合同（字典） 模型准确率标准（字典）

        # 将产生的合同内容记录到智能合约中
        txs = []
        icontractType = 0
        for ic, im in zip( iContracts.values(), iModelLine.values() ):
            icontractType = icontractType + 1
            fee = ic[0]
            reward = ic[1]
            modelLine = im
            tx = self._contract.icontractRecord(icontractType, fee, reward, modelLine)
            txs.append(tx)
        # 将产生的合同类型总数目记录到智能合约
        tx = self._contract.numIcontractTypeRecord(icontractType)
        txs.append(tx)
        return txs

    @methodtools.lru_cache()
    def _evaluate_one_round(self, training_round):  # 评估指定轮的各个参与方本地模型，判断是否达到签订合同预定的准确率标准，并以此为依据为各个参与方记录应获得的奖励数目
        reward = {}  #记录各个参与方本地模型hash与其对应获得的奖励
        for i in range(1, self.clients_num+1):
            # 获取参与方的msg.sender（address）
            address = self._contract.getTrainerMsgSender(i)
            print(f"************address{address}")
            # 获取给定轮和指定trainer的签订的合同类型
            ICType = self._contract.getTrainerICType(address, training_round)
            # 根据合同类型和轮数获取模型准确率标准
            ICModelLine = self._contract.getICModelLine(ICType, training_round)  # 整数 ****
            # 获取给定轮和指定trainer的模型hash
            modelHash_list = self._contract.getTrainerHash(address, training_round)
            modelHash = modelHash_list[0]  # modelHash_list中只有一个元素，因为只获取了一个
            # 根据hash获取模型 _get_models（运行之后可以对其进行修改）
            model_list = self._get_models(modelHash_list)
            model = model_list[0]  # model_list中只有一个元素 modelHash_list中只有一个元素

            # 测试用电量预测模型，计算其误差   !!!!
            model_test_result = 1-self.elec_test(model)  # 1-rmse
            print(f"第{training_round}轮第{i}个参与方本地模型测试得到的结果为{model_test_result}")
            # 与签订合同的标准做对比，得到为其奖励的数值
            if (model_test_result >= ICModelLine / 10000):
                r = self._contract.getICReward(ICType, training_round)  # 获取该合同对应的奖励数目
                reward[modelHash] = r  # 记录应该获得的奖励数目
            else:
                reward[modelHash] = 0  # 记录应该获得的奖励数目
            print(f"reward:{reward[modelHash]}")
        return reward
