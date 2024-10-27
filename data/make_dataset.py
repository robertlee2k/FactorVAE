"""
Generating data using Qlib
Alpha158 is an off-the-shelf dataset provided by Qlib.
"""

import qlib
import pandas as pd
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.contrib.data.handler import Alpha158
import argparse


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='C:/20-python/InvariantStock/data/qlib_data/cn_data') #"/home/bo.li/.qlib/qlib_data/cn_data")
    parser.add_argument("--freq", type=str, default="day")
    parser.add_argument('--start_time', type=str, default='2008-01-01')
    parser.add_argument('--end_time', type=str, default='2024-12-31')
    parser.add_argument('--fit_start_time', type=str, default='2009-01-01')
    parser.add_argument('--fit_end_time', type=str, default='2022-12-31')
    parser.add_argument('--val_start_time', type=str, default='2023-01-01')
    parser.add_argument('--val_end_time', type=str, default='2023-12-31')
    parser.add_argument('--test_start_time', type=str, default='2024-01-01')
    parser.add_argument('--seq_len', type=int, default=61)
    args = parser.parse_args()

    # 根据数据路径初始化Qlib并设置基准和市场
    # Qlib是一个用于量化研究的开源库
    if args.data_path.split('/')[-1] == "cn_data":
        # 如果数据路径指向中国数据，则初始化Qlib为中国市场数据
        # REG_CN是中国市场的区域代码
        qlib.init(provider_uri=args.data_path, region=REG_CN)
        # SH000300是沪深300指数的代码
        benchmark = "SH000300"
        # csi300代表沪深300指数成分股
        market ="csi800"  # "csi300"
    elif args.data_path.split('/')[-1] == "us_data":
        # 如果数据路径指向美国数据，则初始化Qlib为美国市场数据
        # REG_US是美国市场的区域代码
        qlib.init(provider_uri=args.data_path, region=REG_US)
        # ^gspc是标普500指数的代码
        benchmark = "^gspc"
        # sp500代表标普500指数成分股
        market = "sp500"

    provider_uri = args.data_path
    print(f"provider_uri: {provider_uri}")
    print(f"freq: {args.freq}")

    # 初始化数据处理配置字典
    data_handler_config = {
        # 设置数据处理的开始时间
        "start_time": args.start_time,
        # 设置数据处理的结束时间
        "end_time": args.end_time,
        # 设置模型拟合的开始时间
        "fit_start_time": args.fit_start_time,
        # 设置模型拟合的结束时间
        "fit_end_time": args.fit_end_time,
        # 指定感兴趣的金融工具或市场
        "instruments": market,
        # 配置推理数据处理流程
        "infer_processors": [
            # {"class" : "FilterCol", "kwargs" : {"fields_group" : "feature"},},
            # 使用健壮的Z分数标准化处理特征字段，并剪裁异常值
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            # 使用填充缺失值处理器处理特征字段中的缺失值
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
        # 配置学习数据处理流程
        "learn_processors": [
            # 使用删除标签中的缺失值处理器
            {"class": "DropnaLabel", },
            # 使用截面排名标准化处理标签字段
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},  # ！从CSZScoreNorm更改为CSRankNorm
        ],
        # 定义标签字段的计算公式
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }

    # Define time segments for training, validation, and testing
    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }

    # Initialize the dataset object with configuration parameters
    dataset = Alpha158(**data_handler_config)

    # Fetch the feature and label data for the local dataset (low frequency)
    dataframe_L = dataset.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # Remove the top level column names for clearer dataframes
    dataframe_L.columns = dataframe_L.columns.droplevel(0)

    # Fetch the feature and label data for the instantaneous dataset (high frequency)
    dataframe_I = dataset.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    # Remove the top level column names for clearer dataframes
    dataframe_I.columns = dataframe_I.columns.droplevel(0)

    # Check if the dataset is for the Chinese market
    if args.data_path.split('/')[-1] == "cn_data":
        # Market information is not included in the dataset
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        # Save the processed Chinese market data
        dataframe_LM.to_pickle('csi_data.pkl')

    elif args.data_path.split('/')[-1] == "us_data":
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        dataframe_LM.to_pickle('sp500_data.pkl')

    print("数据准备完成，抽样测试")

    ## TEST ##
    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }

    handler = DataHandlerLP.from_df(dataframe_LM)
    dic =  {
            'train': ("2009-01-01", "2019-06-30"),
            'valid': ("2019-07-01", "2019-12-31",),
            'test': ("2020-01-01", "2022-12-31",),
        }
    QlibTSDatasetH = TSDatasetH(handler=handler, segments=dic, step_len=args.seq_len)
    temp = QlibTSDatasetH.prepare(segments="train", data_key=DataHandlerLP.DK_L)

    print("------------------ Test QlibTSDatasetH ------------------")
    print(next(iter(temp)))

    print("程序正常结束")
