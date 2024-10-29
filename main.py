import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

import wandb
from dataset import init_data_loader
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from train_model import train, validate
from utils import set_seed, DataArgs, ModelStructureArgs, ModelManager
from predict import predict_and_eval

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

    model_manager = ModelManager()
    # Start Training
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, args)
        val_loss = validate(factorVAE, valid_dataloader, args)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_file = f'{args.run_name}_factor_{args.num_factor}_hdn_{args.hidden_size}_port_{args.num_portfolio}_seed_{args.seed}.pt'
            model_manager.save_best_model(save_dir=args.save_dir, model_save_file=save_file, model=factorVAE,
                                          loss=best_val_loss)

        if args.wandb:
            wandb.log(
                {"Train Loss": train_loss, "Validation Loss": val_loss, "Learning Rate": scheduler.get_last_lr()[0]})

    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()
    # 进行预测
    print("开始预测阶段....")
    predict_and_eval()

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')
    # 从数据参数存储中加载参数，并将其添加到解析器中
    data_args = DataArgs()
    parser.add_argument("--qlib_data_path", type=str, default=data_args.qlib_data_path,
                        help=data_args.get_help('qlib_data_path'))
    parser.add_argument("--dataset_path", type=str, default=data_args.dataset_path,
                        help=data_args.get_help('dataset_path'))
    parser.add_argument("--save_dir", type=str, default=data_args.save_dir,
                        help=data_args.get_help('save_dir'))
    parser.add_argument("--freq", type=str, default=data_args.freq,
                        help=data_args.get_help('freq'))
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
    parser.add_argument("--normalize", type=bool, default=data_args.normalize,
                        help=data_args.get_help('normalize'))
    parser.add_argument("--select_feature", type=str, default=data_args.select_feature,
                        help=data_args.get_help('select_feature'))
    parser.add_argument("--num_workers", type=int, default=data_args.num_workers,
                        help=data_args.get_help('num_workers'))

    # 从模型结构参数存储中加载参数，并将其添加到解析器中
    model_args = ModelStructureArgs()
    parser.add_argument("--seed", type=int, default=model_args.seed,
                        help=model_args.get_help('seed'))
    parser.add_argument("--run_name", type=str, default=model_args.run_name,
                        help=model_args.get_help('run_name'))
    parser.add_argument("--num_epochs", type=int, default=model_args.num_epochs,
                        help=model_args.get_help('num_epochs'))
    parser.add_argument("--lr", type=float, default=model_args.lr,
                        help=model_args.get_help('lr'))
    parser.add_argument("--num_latent", type=int, default=model_args.num_latent,
                        help=model_args.get_help('num_latent'))
    parser.add_argument("--num_portfolio", type=int, default=model_args.num_portfolio,
                        help=model_args.get_help('num_portfolio'))
    parser.add_argument("--num_factor", type=int, default=model_args.num_factor,
                        help=model_args.get_help('num_factor'))
    parser.add_argument("--hidden_size", type=int, default=model_args.hidden_size,
                        help=model_args.get_help('hidden_size'))
    parser.add_argument("--seq_len", type=int, default=model_args.seq_len,
                        help=model_args.get_help('seq_len'))
    parser.add_argument("--wandb", action='store_false', default=model_args.wandb,
                        help=model_args.get_help('wandb'))

    # 解析命令行参数
    args = parser.parse_args()
    # 打印所有参数取值
    print("所有运行参数取值:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    main(args)
