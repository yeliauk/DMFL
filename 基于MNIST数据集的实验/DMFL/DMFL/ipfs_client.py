import io

import ipfshttpclient
import torch

class IPFSClient:
    # class 属性，以便同一台机器上的 IPFSClient 都可以使用
    _cached_models = {}

    def __init__(self, model_constructor):
        self._model_constructor = model_constructor

    def get_model(self, model_cid):  #通过model_cid(Hash)获取模型参数，返回模型
        model = self._model_constructor()
        if model_cid in self._cached_models:  # 如果_cached_models中有要找的model_cid（Hash)对应的模型参数，那么就直接从_cached_models中拷贝
            # make a deep copy from cache
            # 从缓存中进行深拷贝
            model.load_state_dict(self._cached_models[model_cid])
        else:  # _cached_models中没有要找的model_cid，那从IPFS进行下载
            # download from IPFS
            with ipfshttpclient.connect() as ipfs:
                model_bytes = ipfs.cat(model_cid)  # 利用model_cid(Hash)从IPFS下载对应的模型参数，存入model_bytes()
            buffer = io.BytesIO(model_bytes)  # BytesIO实现了在内存中读写bytes  此处可以理解为向内存中写入model_bytes
            model.load_state_dict(torch.load(buffer))  # 将读取的模型参数加载到模型中
            self._cached_models[model_cid] = model.state_dict()  # 把从IPFS获取的model_cid（Hash)对应的模型参数存入_cached_models
        return model  # 返回模型

    def add_model(self, model):  # 传入模型，将模型参数存储到IPFS，IPFS返回一个对应的Hash值
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)  # 将模型参数保存到buffer
        buffer.seek(0) # ???
        with ipfshttpclient.connect() as ipfs:
            model_cid = ipfs.add_bytes(buffer.read())  # 将buffer的值存入IPFS，IPFS返回Hash值给model_cid
        return model_cid


# model.state_dict() 返回的是一个字典，保存模型参数（模型权重weight 偏置bias）
# model.save(model.state_dict(), buffer) 将模型参数保存到buffer中
# torch.load(buffer) 读取保存的模型参数（与model.save是对应的）
# model.load_state_dict(state_dict)将模型参数加载到模型中
