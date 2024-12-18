import os
import random
from module import *
import pandas as pd
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CommonArgs:
    def get_help(self, field_name):
        """Get help message for a field."""
        return self.help.get(field_name, '')


class DataArgs(CommonArgs):
    # 检查操作系统类型
    is_windows = os.name == 'nt'
    default_qlib_data_path = (
        '../InvariantStock/data/qlib_data/cn_data' if is_windows else
        '../.qlib/qlib_data/cn_data'
    )

    def __init__(self):
        self.qlib_data_path = DataArgs.default_qlib_data_path
        self.dataset_path = './data/csi_data.pkl'
        self.freq = 'day'
        self.save_dir = './best_models'
        # CSI300, CSI800,CSI1000
        self.market = "csi800"
        # data split args
        # "2014-01-01" for csi1000 , "2008-01-01" ,for CSI300 or 800
        self.data_start_time = "2008-01-01"
        # "2015-01-01" for csi1000 "2009-01-01" for CSI300 or 800
        self.fit_start_time = "2009-01-01"
        self.fit_end_time = "2022-12-31"
        self.val_start_time = '2023-01-01'
        self.val_end_time = '2023-12-31'
        self.test_start_time = '2024-01-01'
        self.data_end_time = '2024-10-25'
        self.target_period = 5
        self.target = "CSZScoreNorm"  # or "CSRankNorm" for ranking learning
        self.normalize = False
        self.select_feature = None
        self.num_workers = 4

    @property
    def help(self):
        return {
            'qlib_data_path': 'directory of qlib data',
            'dataset_path': "history dataset prepared for training",
            'freq': "history data frequency",
            'save_dir': 'directory to save model',
            'market': 'stock range indicated by market(csi300,800,1000)',
            'data_start_time': "all data start time (need some buff to do calculations fields like MA60",
            'data_end_time': "all data end time",
            'fit_start_time': "training start_time",
            'fit_end_time': "fit_end_time",
            'val_start_time': "val_start_time",
            'val_end_time': "val_end_time",
            'test_start_time': "test_start_time",
            'target_period': "forecast target period",
            'target':"to learn score or to learn rank",
            'normalize': "whether to normalize the data",
            'select_feature': "select specific feature",
            'num_workers': "number of workers for dataloader",
        }


class ModelStructureArgs(CommonArgs):
    def __init__(self,market):
        super().__init__()
        # Model structure args
        self.num_latent = 158  #alpha158 or alpha360 ( 360 is not useful in this model)
        self.hidden_size = 64
        self.num_factor = 96
        # 1000 for csi1000  # 800 for csi800
        # 从 market 参数中提取数字
        self.num_portfolio = int(''.join(filter(str.isdigit, market)))
        self.seq_len = 60
        # training args
        self.run_name = 'FactorVAE'
        self.num_epochs: int = 20
        self.lr: float = 0.0001
        self.seed: int = 88
        self.wandb: bool = True

    @property
    def help(self):
        return {
            'run_name': 'Name of the run for identification.',
            'num_latent': 'Number of latent variables.',
            'hidden_size': 'Size of the hidden layer.',
            'num_factor': 'Number of factors in the model.',
            'seq_len': "sequence length",
            'num_portfolio': 'number of stocks.',
            'num_epochs': 'number of epochs to train for.',
            'lr': 'Learning rate for the optimizer.',
            'seed': 'Random seed for reproducibility.',
            'wandb': 'Whether to use wandb for logging.',
        }


class ModelManager:
    def __init__(self):
        self.csv_path = 'best_model.csv'

    def save_best_model(self, save_dir, model_save_file, model, loss, epoch):
        save_full_file = os.path.join(save_dir, model_save_file)
        torch.save(model.state_dict(), save_full_file)
        # 更新 best_model.csv
        model_info = {
            'epoch': [epoch],
            'model_save_file': [save_full_file],
            'loss': [loss]
        }
        df = pd.DataFrame(model_info)
        csv_full_file = os.path.join(save_dir, self.csv_path)
        df.to_csv(csv_full_file, index=False, header=True, encoding='utf8')
        print(f"Model saved at {save_full_file}")

    def get_best_model_file(self, save_dir):
        csv_full_file = os.path.join(save_dir, self.csv_path)
        if not os.path.exists(csv_full_file):
            raise FileNotFoundError(f"CSV file {csv_full_file} does not exist.")
        df = pd.read_csv(csv_full_file, encoding='utf8')
        predictor_root = df['model_save_file'].iloc[0]
        return predictor_root

    def setup_model(self, args):
        feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)

        factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio,
                                       hidden_size=args.hidden_size)
        alpha_layer = AlphaLayer(args.hidden_size)
        beta_layer = BetaLayer(args.hidden_size, args.num_factor)

        factor_decoder = FactorDecoder(alpha_layer, beta_layer)
        factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
        factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor, args)
        return factorVAE



