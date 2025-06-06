import functools
import logging
import os
import sys

import yaml


class StringConcatinator(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = '!join'
    @classmethod
    def from_yaml(cls, loader, node):
        return functools.reduce(lambda a, b: a.value + b.value, node.value)


def load_configure(configure_file,type_param):
    '''

    :param configure_filw: 配置文件路径
    :param step: 'Dis' or 'Param' 指定可以进行单独的模型训练，未指定则进行综合训练
    :param type: 模型类型，只读取在配置文件中设置的模型，如需其他模型，自行添加
    :return:
    '''

    if not os.path.exists(configure_file):
        logging.error('config file not exists')
        sys.exit()
        # writer = None
    with open(configure_file, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
            logging.info("loading config file successfully!!")
        except yaml.YAMLError as exc:
            logging.error(exc)

    configs_param = configs['Param']
    # 加载Param配置
    configs_param['normal'] = configs['Param']['Net']['normal']
    configs_param['device'] = configs['device']
    param=type_param
    configs_param['type']=type_param
    if param is not None:
        # Create output folder
        if param == 'VAE':
            values = configs_param['Net']['VAE_Net']
        elif param in ['ODE_RNN']:
            values = configs_param['Net']['ODE_RNN_Net']
        elif param in ['RNN']:
            values = configs_param['Net']['RNN_Net']
    if param is None or param =='MLP':
        values=configs_param['Net']['MLP_Net']
    configs_param['Net'].update(values)
    del configs_param['Net']['ODE_RNN_Net']
    del configs_param['Net']['RNN_Net']
    del configs_param['Net']['MLP_Net']
    del configs_param['Net']['VAE_Net']

    return configs_param
