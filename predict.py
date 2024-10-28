import argparse

import pandas as pd
import qlib
import torch
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq
from torch.xpu import device
from tqdm.auto import tqdm

from dataset import init_data_loader
from utils import ModelStructureArgs, DataArgs, RankIC, ModelManager


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


def predict_on_test(args):
    model_manager = ModelManager()
    factorVAE = model_manager.setup_model(args)
    model_name = model_manager.get_best_model_file(args.save_dir)
    # './best_models/FactorVAE_factor_96_hdn_64_port_800_seed_88.pt'
    # VAE-Revision_factor_64_hdn_32_port_100_seed_42.pt'
    factorVAE.load_state_dict(torch.load(model_name,map_location=torch.device(args.my_device)))
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
    RankIC(predictions, column1='score', column2='LABEL0')
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
    report_normal_df, positions_normal = eval_prediction(predictions, args)
    report_normal_df.to_excel('data/report_normal_df.xlsx')
    positions_normal.to_excel('data/positions_normal.xlsx')


if __name__ == '__main__':
    predict_and_eval()
