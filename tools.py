from itertools import repeat


def get_config_dict(a_class):
    config = {}
    for attr_name, attr_value in a_class.__dict__.items():
        if not attr_name.startswith('__'):  # 排除内置属性
            config[attr_name] = attr_value
    return config


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1
