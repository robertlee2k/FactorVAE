"""
Generating data using Qlib
Alpha158 is an off-the-shelf dataset provided by Qlib.
"""

import qlib
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import TSDatasetH
from qlib.contrib.data.handler import Alpha158, Alpha360
import argparse
from utils import DataArgs,ModelStructureArgs

if __name__ == "__main__":
    # 创建一个数据参数实例
    data_args = DataArgs()

    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--qlib_data_path", type=str, default=data_args.qlib_data_path,
                        help=data_args.get_help('qlib_data_path'))
    parser.add_argument("--save_dir", type=str, default=data_args.save_dir,
                        help=data_args.get_help('save_dir'))
    parser.add_argument("--market", type=str, default=data_args.market,
                        help=data_args.get_help('market'))
    parser.add_argument("--data_start_time", type=str, default=data_args.data_start_time,
                        help=data_args.get_help('data_start_time'))
    parser.add_argument("--data_end_time", type=str, default=data_args.data_end_time,
                        help=data_args.get_help('data_end_time'))
    parser.add_argument("--fit_start_time", type=str, default=data_args.fit_start_time,
                        help=data_args.get_help('fit_start_time'))
    parser.add_argument("--fit_end_time", type=str, default=data_args.fit_end_time,
                        help=data_args.get_help('fit_end_time'))
    parser.add_argument("--val_start_time", type=str, default=data_args.val_start_time,
                        help=data_args.get_help('val_start_time'))
    parser.add_argument("--val_end_time", type=str, default=data_args.val_end_time,
                        help=data_args.get_help('val_end_time'))
    parser.add_argument("--test_start_time", type=str, default=data_args.test_start_time,
                        help=data_args.get_help('test_start_time'))
    parser.add_argument("--target_period", type=str, default=data_args.target_period,
                        help=data_args.get_help('target_period'))
    parser.add_argument("--target", type=str, default=data_args.target,
                        help=data_args.get_help('target'))
    parser.add_argument("--normalize", type=bool, default=data_args.normalize,
                        help=data_args.get_help('normalize'))
    parser.add_argument("--select_feature", type=str, default=data_args.select_feature,
                        help=data_args.get_help('select_feature'))
    parser.add_argument("--dataset_path", type=str, default=data_args.dataset_path,
                        help=data_args.get_help('dataset_path'))
    parser.add_argument("--freq", type=str, default=data_args.freq,
                        help=data_args.get_help('freq'))

    # 创建模型参数实例 （num_latent 和 seq_len 有用）
    model_args = ModelStructureArgs(data_args.market)
    parser.add_argument("--num_latent", type=int, default=model_args.num_latent,
                        help=model_args.get_help('num_latent'))
    parser.add_argument("--seq_len", type=int, default=model_args.seq_len,
                        help=model_args.get_help('seq_len'))

    args = parser.parse_args()
    # 打印所有参数取值
    print("所有运行参数取值:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # 根据数据路径初始化Qlib并设置基准和市场
    # Qlib是一个用于量化研究的开源库
    provider_uri = args.qlib_data_path
    region = None
    print(f"provider_uri: {provider_uri}")
    if provider_uri.split('/')[-1] == "cn_data":
        # 如果数据路径指向中国数据，则初始化Qlib为中国市场数据
        # REG_CN是中国市场的区域代码
        region = REG_CN
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        # SH000300是沪深300指数的代码
        # benchmark = "SH000300"
        # csi300代表沪深300指数成分股
        market = args.market #"csi1000" "csi800"  # "csi300"
    elif provider_uri.split('/')[-1] == "us_data":
        # 如果数据路径指向美国数据，则初始化Qlib为美国市场数据
        # REG_US是美国市场的区域代码
        region = REG_US
        qlib.init(provider_uri=provider_uri, region=REG_US)
        # ^gspc是标普500指数的代码
        # benchmark = "^gspc"
        # sp500代表标普500指数成分股
        market = "sp500"
    else:
        print("Invalid qlib data path.")
        exit(1)

    # 标签字段的计算公式,目标设为预测target_period日收益率

    label_target = f"Ref($close, -{args.target_period})/Ref($close, -1) - 1"
    print("标签计算公式为：", label_target)
    # 初始化数据处理配置字典
    data_handler_config = {
        # 设置数据处理的开始时间
        "start_time": args.data_start_time,
        # 设置数据处理的结束时间
        "end_time": args.data_end_time,
        # 设置模型拟合的开始时间
        "fit_start_time": args.fit_start_time,
        # 设置模型拟合的结束时间
        "fit_end_time": args.fit_end_time,
        # 指定感兴趣的金融工具或市场
        "instruments": market,
        # 配置推理数据处理流程
        "infer_processors": [
            # {"class" : "FilterCol", "kwargs" : {"fields_group" : "feature"},},
            # 使用健壮的Z分数标准化处理特征字段，并剪裁异常值--removed by me,目前都是用alpha158因子，这些都已经做了必要norm，决定不用
            #{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            # 使用填充缺失值处理器处理特征字段中的缺失值
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
        # 配置学习数据处理流程
        "learn_processors": [
            # 使用删除标签中的缺失值处理器
            {"class": "DropnaLabel", },
            ## removed by me: 保持训练推理时同样逻辑一致，使用健壮的Z分数标准化处理特征字段，并剪裁异常值
            #{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            # 使用截面排名标准化处理标签字段 ，！从CSRankNorm更改为CSZScoreNorm，修改了损失函数，目标直接预测涨幅
            {"class": args.target, "kwargs": {"fields_group": "label"}},
        ],
        # 定义标签字段的计算公式
        "label": [label_target],
    }

    # Initialize the dataset object with configuration parameters
    if args.num_latent == 158:
        dataset = Alpha158(**data_handler_config)
    elif args.num_latent == 360:
        dataset = Alpha360(**data_handler_config)
    else:
        dataset = None

    # Fetch the feature and label data for the local dataset (low frequency)
    dataframe_L = dataset.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # Remove the top level column names for clearer dataframes
    dataframe_L.columns = dataframe_L.columns.droplevel(0)

    # Fetch the feature and label data for the instantaneous dataset (high frequency)
    dataframe_I = dataset.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    # Remove the top level column names for clearer dataframes
    dataframe_I.columns = dataframe_I.columns.droplevel(0)

    # Check if the dataset is for the Chinese market
    if region == REG_CN:
        # Market information is not included in the dataset
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        # Save the processed Chinese market data
        dataframe_LM.to_pickle(args.dataset_path)

    elif region == REG_US:
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        dataframe_LM.to_pickle('sp500_data.pkl')

    print("数据保存完成")
    # 打印高频数据集的行数和列数
    print(f"高频数据集的形状: {dataframe_I.shape}")

    # 重置索引，将复合索引转换为普通列
    dataframe_L_reset = dataframe_L.reset_index()
    # 按 datetime 分组并计算每组的 instrument 数量
    grouped_L = dataframe_L_reset.groupby('datetime')['instrument'].nunique()

    # 找到记录最多和最少的一天
    max_date_L = grouped_L.idxmax()
    max_count_L = grouped_L.max()
    min_date_L = grouped_L.idxmin()
    min_count_L = grouped_L.min()
    # 获取低频数据集的最早和最晚日期
    earliest_date_L = dataframe_L_reset['datetime'].min()
    latest_date_L = dataframe_L_reset['datetime'].max()

    # 打印低频数据集的信息
    # 打印低频数据集的行数和列数
    print(f"低频数据集的形状: {dataframe_L.shape}")
    print(f"低频数据集的最早日期: {earliest_date_L}")
    print(f"低频数据集的最晚日期: {latest_date_L}")
    print(f"低频数据集中记录最多的一天: {max_date_L}，instrument 数量: {max_count_L}")
    print(f"低频数据集中记录最少的一天: {min_date_L}，instrument 数量: {min_count_L}")

    print("数据测试")

    ## TEST ##
    # Define time segments for training, validation, and testing
    segments = {
        'train': (args.fit_start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.data_end_time)
    }

    handler = DataHandlerLP.from_df(dataframe_LM)

    QlibTSDatasetH = TSDatasetH(handler=handler, segments=segments, step_len=args.seq_len) # 这里只是做个测试，
    temp = QlibTSDatasetH.prepare(segments="train", data_key=DataHandlerLP.DK_L)

    print("------------------ Test QlibTSDatasetH 训练集的第一个iterate------------------")
    print(next(iter(temp)))

    print("程序正常结束")
