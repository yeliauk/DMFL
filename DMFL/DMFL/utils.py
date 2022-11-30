def print_global_performance(client):  #  打印某个client全局模型的表现
    loss = client.evaluate_current_global()
    print(f"\t\t{client.name} got: Loss {loss}")


def print_token_count(client):  #打印某个client token的数目 总共token数目 以及占比
    tokens = client.get_token_count()
    total_tokens = client.get_total_token_count()
    percent = int(100*tokens/total_tokens) if tokens > 0 else 0
    print(f"\t\t{client.name} has {tokens} of {total_tokens} tokens ({percent}%)")
