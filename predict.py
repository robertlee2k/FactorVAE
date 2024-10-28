import argparse
import numpy as np
import pandas as pd
import qlib
import torch
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from dataset import init_data_loader
from utils import ModelStructureArgs, DataArgs, ModelManager


def load_predict_args():
    parser = argparse.ArgumentParser(description='Predict using trained model on stock data')
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--my_device", type=str, default=my_device, help="device to use")

    # 从数据参数存储中加载参数，并将其添加到解析器中
    data_args = DataArgs()
    parser.add_argument("--qlib_data_path", type=str, default=data_args.qlib_data_path,
                        help=data_args.get_help('qlib_data_path'))
    parser.add_argument("--dataset_path", type=str, default=data_args.dataset_path,
                        help=data_args.get_help('dataset_path'))
    parser.add_argument("--save_dir", type=str, default=data_args.save_dir,
                        help=data_args.get_help('save_dir'))
    parser.add_argument("--seq_len", type=int, default=data_args.seq_len,
                        help=data_args.get_help('seq_len'))
    parser.add_argument("--test_start_time", type=str, default=data_args.test_start_time,
                        help=data_args.get_help('test_start_time'))
    parser.add_argument("--data_end_time", type=str, default=data_args.data_end_time,
                        help=data_args.get_help('data_end_time'))
    # 从模型结构参数存储中加载参数，并将其添加到解析器中
    model_args = ModelStructureArgs()
    parser.add_argument("--run_name", type=str, default=model_args.run_name,
                        help=model_args.get_help('run_name'))
    parser.add_argument("--num_latent", type=int, default=model_args.num_latent,
                        help=model_args.get_help('num_latent'))
    parser.add_argument("--num_portfolio", type=int, default=model_args.num_portfolio,
                        help=model_args.get_help('num_portfolio'))
    parser.add_argument("--num_factor", type=int, default=model_args.num_factor,
                        help=model_args.get_help('num_factor'))
    parser.add_argument("--hidden_size", type=int, default=model_args.hidden_size,
                        help=model_args.get_help('hidden_size'))
    final_args = parser.parse_args()
    # 打印所有参数取值
    print("所有运行参数取值:")
    for arg in vars(final_args):
        print(f"{arg}: {getattr(final_args, arg)}")
    return final_args


@torch.no_grad()
def generate_prediction_scores(model, test_dataloader, args):
    device = args.my_device
    print(device)
    model.to(device)
    model.eval()
    ls = []

    with tqdm(total=len(test_dataloader)) as pbar:  # -args.seq_length+1
        for i, (char_with_label, _) in enumerate(test_dataloader):
            char = char_with_label[:, :, :-1].to(device)
            if char.shape[1] != args.seq_len:
                print(f"输入序列长度不是{args.seq_len}，跳过")
                continue
            predictions = model.prediction(char.float())
            ls.append(predictions.detach().cpu())
            pbar.update(1)

    ls = torch.cat(ls, dim=0)
    indexes = test_dataloader.dataset.sampler.get_index()
    multi_index = pd.MultiIndex.from_tuples(indexes, names=["datetime", "instrument"])
    ls = pd.DataFrame(ls.numpy(), index=multi_index, columns=['score'])
    return ls


def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.mean(ric_values_multiindex)
    std = np.std(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})


def predict_on_test(args):
    model_manager = ModelManager()
    factorVAE = model_manager.setup_model(args)
    model_name = model_manager.get_best_model_file(args.save_dir)
    # './best_models/FactorVAE_factor_96_hdn_64_port_800_seed_88.pt'
    # VAE-Revision_factor_64_hdn_32_port_100_seed_42.pt'
    factorVAE.load_state_dict(torch.load(model_name, map_location=torch.device(args.my_device)))
    dataset = pd.read_pickle(args.dataset_path)  # .iloc[:, :159]
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True)
    test_dataloader = init_data_loader(dataset,
                                       shuffle=False,
                                       step_len=args.seq_len,
                                       start=args.test_start_time,
                                       end=args.data_end_time)
    output = generate_prediction_scores(factorVAE, test_dataloader, args)
    output = pd.merge(output, dataset['LABEL0'], right_index=True, left_index=True)
    print(output.reset_index())
    # output.to_pickle('data/predictions.pkl')
    return output


def eval_prediction(predictions, args):
    # init qlib
    qlib.init(provider_uri=args.qlib_data_path)
    CSI300_BENCH = "SH000300"
    FREQ = "day"
    STRATEGY_CONFIG = {
        "topk": 20,
        "n_drop": 2,
        # pred_score, pd.Series
        "signal": predictions,
    }
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    backtest_config = {
        "start_time": args.test_start_time,
        "end_time": args.data_end_time,
        "account": 10000000,
        "benchmark": CSI300_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }
    # strategy object
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    # backtest info
    report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)
    return report_normal_df, positions_normal


def predict_and_eval():
    args = load_predict_args()
    predictions = predict_on_test(args)
    predictions.to_excel("data/predections.xlsx")
    print("输出预测的 rankic 结果：")
    rankic_df = RankIC(predictions, column1='score', column2='LABEL0')
    print(rankic_df)

    report_normal_df, positions_normal = eval_prediction(predictions, args)
    report_normal_df.to_excel('data/report_normal_df.xlsx')
    # general_df, stock_pos_df = process_qlib_position(positions_normal)
    # general_df.to_excel('data/general_df.xlsx')
    # stock_pos_df.to_excel('data/stock_pos_df.xlsx')


# position_normal 是个嵌套dict形状
def process_qlib_position(positions):
    # 创建空列表来存储总体信息和个股详情
    general_info_list = []
    stock_details_list = []


    for timestamp, daily_obj in positions.items():
        #  daily_obj 是这个形状：{'_settle_type': 'None', 'position': {'cash': 10000000, 'now_account_value': 10000000.0}, 'init_cash': 10000000}
        print(f"交易日期：{timestamp}")
        settle_type = daily_obj['_settle_type']
        print(f"交易类型：{settle_type}")
        pos = daily_obj['position']
        print(f"持仓情况：{pos}")
        # 获取总体信息
        general_info = {
            'timestamp': timestamp,
            'cash': pos['cash'],
            'now_account_value': pos['now_account_value']
        }
        general_info_list.append(general_info)

        # 获取个股详情
        for stock_code, details in pos.items():
            stock_details = {
                'timestamp': timestamp,
                'stock_code': stock_code,
                'amount': details.get('amount'),
                'weight': details.get('weight')
            }
            stock_details_list.append(stock_details)

    # 创建 DataFrame
    general_df = pd.DataFrame(general_info_list)
    stock_pos_df = pd.DataFrame(stock_details_list)

    return general_df, stock_pos_df


if __name__ == '__main__':
    predict_and_eval()
