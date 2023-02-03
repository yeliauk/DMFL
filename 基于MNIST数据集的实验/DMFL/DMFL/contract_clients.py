import json

import base58
from web3 import HTTPProvider, Web3

class BaseEthClient:

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    NETWORK_ID = "5777"

    def __init__(self, account_idx):
        self._w3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))

        self.address = self._w3.eth.accounts[account_idx]
        self._w3.eth.defaultAccount = self.address

        self.txs = []
    
    def wait_for_tx(self, tx_hash):
        receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)  
        return receipt

    def get_gas_used(self):
        receipts = [self._w3.eth.getTransactionReceipt(tx) for tx in self.txs]  # getTransactionReceipt返回指定交易的收据对象
        gas_amounts = [receipt['gasUsed'] for receipt in receipts]
        return sum(gas_amounts)

class _BaseContractClient(BaseEthClient):

    IPFS_HASH_PREFIX = bytes.fromhex('1220')  # IPFS hash的前缀

    def __init__(self, contract_json_path, account_idx, contract_address, deploy):
        super().__init__(account_idx)

        self._contract_json_path = contract_json_path  # 智能合约编译之后生成的json文件路径

        self._contract, self.contract_address = self._instantiate_contract(contract_address, deploy)

    def _instantiate_contract(self, address=None, deploy=False):  #  实例化合约
        with open(self._contract_json_path) as json_file:
            crt_json = json.load(json_file)
            abi = crt_json['abi']
            bytecode = crt_json['bytecode']
            if address is None:
                if deploy:
                    tx_hash = self._w3.eth.contract(
                        abi=abi,
                        bytecode=bytecode
                    ).constructor().transact()  # tx_hash
                    self.txs.append(tx_hash)  # 将tx_hash添加到self.txs[]中
                    tx_receipt = self.wait_for_tx(tx_hash)
                    address = tx_receipt.contractAddress
                else:
                    address = crt_json['networks'][self.NETWORK_ID]['address']
        instance = self._w3.eth.contract(
            abi=abi,
            address=address
        )
        return instance, address

    def _to_bytes32(self, model_cid):
        #  base58和base64一样是一种二进制转可视字符串的算法，主要用来转换大整数值。区别是，转换出来的字符串，
        #  去除了几个看起来会产生歧义的字符，如 0 (零), O (大写字母O), I (大写的字母i) and l (小写的字母L) ，
        #  和几个影响双击选择的字符，如/, +。结果字符集正好58个字符(包括9个数字，24个大写字母，25个小写字母)。
        #  不同的应用实现中，base58 最后查询的字母表可能不同，所以没有具体的标准。
        #  比特币地址123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
        #  详细可见https://blog.csdn.net/drift_along/article/details/122018240
        bytes34 = base58.b58decode(model_cid)
        assert bytes34[:2] == self.IPFS_HASH_PREFIX, \
            f"IPFS cid should begin with {self.IPFS_HASH_PREFIX} but got {bytes34[:2].hex()}"
        bytes32 = bytes34[2:]
        return bytes32

    def _from_bytes32(self, bytes32):
        bytes34 = self.IPFS_HASH_PREFIX + bytes32
        model_cid = base58.b58encode(bytes34).decode()
        return model_cid


class ContractClient(_BaseContractClient):
    def __init__(self, account_idx, address, deploy):
        super().__init__(
            "build/contracts/DMFL.json",
            account_idx,
            address,
            deploy
        )

    def evaluator(self):  # 返回评估者的地址
        return self._contract.functions.evaluator().call()

    def genesis(self):  # 返回genesis模型的hash
        cid_bytes = self._contract.functions.genesis().call()
        return self._from_bytes32(cid_bytes)

    def updates(self, training_round):  # 返回指定训练轮的各个参与方模型参数hash [*** *** ... ***]
        cid_bytes = self._contract.functions.updates(training_round).call()
        return [self._from_bytes32(b) for b in cid_bytes]

    def currentRound(self):  # 返回当前训练轮数 uint256
        return self._contract.functions.currentRound().call()

    def secondsRemaining(self):  # 返回当前训练回合剩余的秒数 uint256
        return self._contract.functions.secondsRemaining().call()

    def countTokens(self, address=None, training_round=None):  # 返回指定地址在指定轮数时获得到的所有token值（如果轮数为空，则默认指定为当前轮数）
        if address is None:
            address = self.address
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTokens(address, training_round).call()

    def countTotalTokens(self, training_round=None):  # 返回指定轮数时所有trainer获得到的所有token值（如果轮数为空，则默认指定为当前轮数）
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTotalTokens(training_round).call()


    def countCurrentToken(self, address=None, training_round=None):  # 返回指定地址在指定轮这一轮获得的奖励数目
        return self._contract.functions.countCurrentToken(address, training_round).call()

    def countCurrentTotalTokens(self, training_round=None):  # 返回指定轮数这一轮所有trainer获得的奖励总数目
        return self._contract.functions.countCurrentTotalTokens(training_round).call()


    def madeContribution(self, address, training_round):  # 指定地址在指定轮中是否作出贡献 true/false
        return self._contract.functions.madecontribution(address, training_round).call()

    def setGenesis(self, model_cid, round_duration, max_num_updates):  # 设置genesis
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes, round_duration, max_num_updates).call()
        tx = self._contract.functions.setGenesis(cid_bytes, round_duration, max_num_updates).transact()
        self.txs.append(tx)
        return tx

    def addModelUpdate(self, model_cid, training_round):  # 记录指定轮的各个参与方模型参数对应的Hash值
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.addModelUpdate(
            cid_bytes, training_round).call()
        tx = self._contract.functions.addModelUpdate(
            cid_bytes, training_round).transact()
        self.txs.append(tx)
        return tx

    def setTokens(self, model_cid, num_tokens):  # 分配奖励
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setTokens(cid_bytes, num_tokens).call()
        tx = self._contract.functions.setTokens(
            cid_bytes, num_tokens).transact()
        self.txs.append(tx)
        return tx



    # 在智能合约上记录发布的合同内容
    def icontractRecord(self, icontractType, registFee, reward, modelLine):
        self._contract.functions.icontractRecord(
            icontractType, registFee, reward, modelLine).call()
        tx = self._contract.functions.icontractRecord(
            icontractType, registFee, reward, modelLine).transact()
        self.txs.append(tx)
        return tx

    # 在智能合约上记录合同类型的数目
    def numIcontractTypeRecord(self, typeNum):
        self._contract.functions.numIcontractTypeRecord(typeNum).call()
        tx = self._contract.functions.numIcontractTypeRecord(typeNum).transact()
        self.txs.append(tx)
        return tx


    # 获取智能合约上某一轮某一类型的合同注册费用
    def getAFee(self, icontractType, round):
        return self._contract.functions.getAFee(icontractType, round).call()


    # 在智能合约上记录参与方每轮合同签订情况（类型、缴纳的注册费）
    def signContract(self, icType, fee):
        self._contract.functions.signContract(icType, fee).call()
        tx = self._contract.functions.signContract(icType, fee).transact()
        self.txs.append(tx)
        return tx


    # 在智能合约上记录参与方的账户地址msg.sender，方便合同发布者获取各个参与方签订的合同
    def trainerMsgSenderRecord(self, index):
        self._contract.functions.trainerMsgSenderRecord(index).call()
        tx = self._contract.functions.trainerMsgSenderRecord(index).transact()
        self.txs.append(tx)
        return tx

    # 获取某个trainer的msg.sender
    def getTrainerMsgSender(self, index):
        return self._contract.functions.getTrainerMsgSender(index).call()

    # 获取某个参与方某轮签订的合同类型
    def getTrainerICType(self, addr, round):
        return self._contract.functions.getTrainerICType(addr, round).call()

    # 获取某一个合同对应的模型准确率标准
    def getICModelLine(self, type, round):
        return self._contract.functions.getICModelLine(type, round).call()

    # 获取某一个参与方某轮的模型hash
    def getTrainerHash(self, addr, round):
        model_hash = self._contract.functions.getTrainerHash(addr, round).call()
        return [self._from_bytes32(model_hash)]

    # 获取某一个合同对应的奖励数目
    def getICReward(self, type, round):
        return self._contract.functions.getICReward(type, round).call()

