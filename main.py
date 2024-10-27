import torch
import pandas as pd
import os
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import init_data_loader
from train_model import train, validate
from utils import set_seed, DataArgument
import wandb


def print_train_batches_info(dataloader):
    try:
        total_train_batches = len(dataloader)
        print(f"总共有 {total_train_batches} 个批次 （实现上是每天作为一个批次）")
        if total_train_batches == 0:
            print("数据加载器为空，没有批次可以处理。")
            return

        for i, batch in enumerate(dataloader):
            if i == 0 or i == (total_train_batches // 2) or i == (total_train_batches - 1):
                print(f"第 {i + 1} 个批次的大小: {len(batch)}")
    except Exception as e:
        print(f"处理训练数据加载器时发生错误: {e}")


def main(args):
    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create model
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio,
                                   hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)

    # create dataloaders
    dataset = pd.read_pickle(args.dataset_path)
    print("数据pkl文件已加载：数据列为")
    for col in dataset.columns:
        print(col)
    dataset = dataset.iloc[:, :159]  # 从指定路径读取数据集，并选择前159列，排除市场信息
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True)  # 将数据集的最后一列重命名为 'LABEL0'，表示预测因子目标。
    print("更名处理后：")
    print(dataset.head())

    train_dataloader = init_data_loader(dataset,
                                        shuffle=True,
                                        step_len=args.seq_len,
                                        start=args.fit_start_time,
                                        end=args.fit_end_time,
                                        select_feature=args.select_feature)

    valid_dataloader = init_data_loader(dataset,
                                        shuffle=False,
                                        step_len=args.seq_len,
                                        start=args.val_start_time,
                                        end=args.val_end_time,
                                        select_feature=args.select_feature)
    # 打印训练和验证数据加载器的批次大小
    print("训练数据加载器的动态批次大小：")
    print_train_batches_info(train_dataloader)
    print("验证数据加载器的动态批次大小：")
    print_train_batches_info(valid_dataloader)

    T_max = len(train_dataloader) * args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************")
    args.device = device

    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if args.wandb:
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
        wandb.config.update(args)

    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, args)
        val_loss = validate(factorVAE, valid_dataloader, args)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # ? save model in save_dir

            # ? torch.save
            save_root = os.path.join(args.save_dir,
                                     f'{args.run_name}_factor_{args.num_factor}_hdn_{args.hidden_size}_port_{args.num_portfolio}_seed_{args.seed}.pt')
            torch.save(factorVAE.state_dict(), save_root)
            print(f"Model saved at {save_root}")

        if args.wandb:
            wandb.log(
                {"Train Loss": train_loss, "Validation Loss": val_loss, "Learning Rate": scheduler.get_last_lr()[0]})

    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()


if __name__ == '__main__':
    # 创建一个数据参数实例
    default_args = DataArgument()

    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

    parser.add_argument("--qlib_data_path", type=str, default=default_args.qlib_data_path,
                        help=default_args.get_help('qlib_data_path'))
    parser.add_argument("--dataset_path", type=str, default=default_args.dataset_path,
                        help=default_args.get_help('dataset_path'))
    parser.add_argument("--freq", type=str, default=default_args.freq,
                        help=default_args.get_help('freq'))
    parser.add_argument("--save_dir", type=str, default=default_args.save_dir,
                        help=default_args.get_help('save_dir'))
    parser.add_argument("--data_start_time", type=str, default=default_args.data_start_time,
                        help=default_args.get_help('data_start_time'))
    parser.add_argument("--data_end_time", type=str, default=default_args.data_end_time,
                        help=default_args.get_help('data_end_time'))
    parser.add_argument("--fit_start_time", type=str, default=default_args.fit_start_time,
                        help=default_args.get_help('fit_start_time'))
    parser.add_argument("--fit_end_time", type=str, default=default_args.fit_end_time,
                        help=default_args.get_help('fit_end_time'))
    parser.add_argument("--val_start_time", type=str, default=default_args.val_start_time,
                        help=default_args.get_help('val_start_time'))
    parser.add_argument("--val_end_time", type=str, default=default_args.val_end_time,
                        help=default_args.get_help('val_end_time'))
    parser.add_argument("--seq_len", type=int, default=default_args.seq_len,
                        help=default_args.get_help('seq_len'))
    parser.add_argument("--normalize", type=bool, default=default_args.normalize,
                        help=default_args.get_help('normalize'))
    parser.add_argument("--select_feature", type=str, default=default_args.select_feature,
                        help=default_args.get_help('select_feature'))

    parser.add_argument('--seed', type=int, default=88, help='random seed')
    parser.add_argument('--run_name', type=str, default='FactorVAE', help='name of the run')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train for') # changed from 30
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_latent', type=int, default=158, help='number of variables')
    parser.add_argument('--num_portfolio', type=int, default=800, help='number of stocks')  # changed from 128
    parser.add_argument('--num_factor', type=int, default=96, help='number of factors')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--wandb', action='store_false', default=True, help='whether to use wandb')

    args = parser.parse_args()
    # 打印所有参数取值
    print("所有运行参数取值:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    main(args)
