import random
from module import *
from tqdm import tqdm
from scipy.stats import spearmanr
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataArgument:
    # 检查操作系统类型
    is_windows = os.name == 'nt'
    default_qlib_data_path = (
        'C:/20-python/InvariantStock/data/qlib_data/cn_data' if is_windows else
        '/home/bo.li/.qlib/qlib_data/cn_data'
    )

    def __init__(self):
        self.qlib_data_path = DataArgument.default_qlib_data_path
        self.dataset_path = './data/csi_data.pkl'
        self.freq = 'day'
        self.save_dir = './best_models'
        self.data_start_time = "2008-01-01"
        self.fit_start_time = "2009-01-01"
        self.fit_end_time = "2022-12-31"
        self.val_start_time = '2023-01-01'
        self.val_end_time = '2023-12-31'
        self.test_start_time = '2024-01-01'
        self.data_end_time = '2024-12-31'
        self.seq_len = 61
        self.normalize = False
        self.select_feature = None

    @property
    def help(self):
        return {
            'qlib_data_path': 'directory of qlib data',
            'dataset_path': "history dataset prepared for training",
            'freq': "history data frequency",
            'save_dir': 'directory to save model',
            'data_start_time': "all data start time (need some buff to do calculations fields like MA60",
            'data_end_time': "all data end time",
            'fit_start_time': "training start_time",
            'fit_end_time': "fit_end_time",
            'val_start_time': "val_start_time",
            'val_end_time': "val_end_time",
            'test_start_time': "test_start_time",
            'seq_len': "sequence length",
            'normalize': "whether to normalize the data",
            'select_feature': "select specific feature",
        }

    def get_help(self, field_name):
        """get help message of a field"""
        return self.help.get(field_name, '')


def load_model(args):
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)

    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio,
                                   hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)

    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    return factorVAE


@torch.no_grad()
def generate_prediction_scores(model, test_dataloader, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    model.eval()
    test_loss = 0
    ls = []

    with tqdm(total=len(test_dataloader)) as pbar:  # -args.seq_length+1
        for i, (char, _) in enumerate(test_dataloader):
            char = char.to(device)
            if char.shape[1] != args.seq_length:
                print("?")
                continue
            predictions = model.prediction(char.float())
            ls.append(predictions.detach().cpu())
            pbar.update(1)

    ls = torch.cat(ls, dim=0)
    multi_index = pd.MultiIndex.from_tuples(test_dataset.get_index(), names=["datetime", "instrument"])
    ls = pd.DataFrame(ls.numpy(), index=multi_index, columns=['score'])
    return ls


class test_args:
    run_name: str
    num_factor: int
    normalize: bool = True
    select_feature: bool = True

    batch_size: int = 300
    seq_length: int = 20

    hidden_size: int = 20
    num_latent: int = 20
    num_portfolio: int = 128

    save_dir = './best_model'
    use_qlib: bool = False


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
