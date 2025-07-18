import time
import numpy as np
import pandas
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from utils import *
from pretrain_models import *
from e2e_models import *
from edcoders import *

seed_list = list(range(3407, 10000, 10))

def pretrain(model, train_dataloader, args):
    print("start pre-training..")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    max_epoch = args.epoch_pretrain
    epoch_iter = tqdm(range(max_epoch))
    device = args.device
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batched_data in train_dataloader:
            batched_graph = batched_data 
            batched_graph = batched_graph.to(device)
            loss, loss_dict = model(batched_graph, batched_graph.ndata["feature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        train_loss = np.mean(loss_list)
        loss_dict["lr"] = get_current_lr(optimizer)
        print(f"# Epoch {epoch} \n\t train_loss: {train_loss:.4f} \n\t {loss_dict}")
    return model

def work(dataset: Dataset, kernel='gcn', cross_mode='ng2ng', args=None):
    time_cost = 0
    pretrain_model_name = args.pretrain_model
    if '-' in kernel:
        encoder_type, decoder_type = kernel.split('-')
    else:
        encoder_type = kernel
        decoder_type=kernel
        if encoder_type == 'bwgnn':
            decoder_type="gcn"
    encoder_name = kernel
    full_model_name = pretrain_model_name + '-' + encoder_name
    hop = args.khop
    dataset_name = dataset.name.replace('/', '.')
    print(f"preparing the dataset for {args.trials} trials")
    dataset.prepare_dataset(total_trials=args.trials)
    print(f"making the subpooling matrix")
    dataset.make_sp_matrix_graph_list(khop=hop, load_kg=(not args.force_remake_sp))

    pretrain_seed = 42
    set_seed(pretrain_seed)
    torch.cuda.empty_cache()

    in_dim = dataset.in_dim
    if args.pretrain_model == 'graphmae':
        pretrain_model = GraphMAE(in_dim=in_dim, hid_dim=args.hid_dim, num_layer=args.num_layer_pretrain,  
                                  drop_ratio=args.dropout, act=args.act, norm=args.norm, residual=args.residual,
                                  mask_ratio=args.mask_ratio, encoder_type=encoder_type, decoder_type=decoder_type,
                                  replace_ratio=args.replace_ratio).to(args.device)
    else:
        raise NotImplementedError
    
    full_model_name += '-' + decoder_type
    model_path = f"../pretrained_models/{full_model_name}_{dataset_name}_{args.epoch_pretrain}.pt"
    if args.load_model == "":
        print(pretrain_model)
        pretrain_dataloader = dataset.get_pretrain_dataloaders(args.batch_size)
        pretrain_model = pretrain(pretrain_model, pretrain_dataloader, args)
        print(f"model saved to {model_path}")
        if not os.path.exists('../pretrained_models/'):
            os.mkdir('../pretrained_models/')
        torch.save(pretrain_model.state_dict(), model_path)
        del pretrain_dataloader
    else:
        pretrain_model.load_state_dict(torch.load(model_path))
    
    torch.cuda.empty_cache()

    result_score_dict_list = []
    for t in range(args.trials):
        print("Dataset {}, Model {}, Trial {}".format(dataset_name, full_model_name, t))
        pretrain_model.load_state_dict(torch.load(model_path))
        seed = seed_list[t]
        set_seed(seed)
        train_dataloader, val_dataloader, test_dataloader = dataset.get_graph_and_sp_dataloaders(args.batch_size, trial_id=t)
        e2e_model = UnifyMLPDetector(pretrain_model, dataset, (train_dataloader, val_dataloader, test_dataloader), cross_mode=cross_mode, args=args)
        ST = time.time()
        print(f"training...")
        score_test = e2e_model.train()
        result_score_dict_list.append(score_test)
        ED = time.time()
        time_cost += ED - ST
    
    model_result = {
        'Dataset': dataset_name,
        'Model': full_model_name,
        'CrossMode': cross_mode,
        'Time(s)': time_cost / args.trials
    }

    for level_key, metrics_dict in result_score_dict_list[0].items(): # Assuming all trials have same keys
        level_name = NAME_MAP.get(level_key, 'Unknown')
        for metric_name in metrics_dict.keys():
            metric_result_list = [d[level_key][metric_name] for d in result_score_dict_list]
            model_result[f'{metric_name}_{level_name}_mean'] = np.mean(metric_result_list)
            model_result[f'{metric_name}_{level_name}_std'] = np.std(metric_result_list)


    save_results_to_csv(model_result, args)
    
    print("--- Experiment Summary ---")
    print(pd.DataFrame([model_result]))
    print("--------------------------")
    
    return pd.DataFrame([model_result])

def main():
    args = get_args()
    if args.kernels is not None:
        kernels = args.kernels.split(',')
        print('All kernels: ', kernels)

    if args.datasets is not None:
        if '-' in args.datasets:
            st, ed = args.datasets.split('-')
            dataset_names = DATASETS[int(st):int(ed)+1]
        else:
            dataset_names = [DATASETS[int(t)] for t in args.datasets.split(',')]
        print('All Datasets: ', dataset_names)

    if args.cross_modes is not None:
        cross_modes = args.cross_modes.split(',')
        print('All Cross_modes: ', cross_modes)

    if args.khop == 1:
        sp_type = "star+norm"
    elif args.khop == 2:
        sp_type = "convtree+norm"
    else:
        raise NotImplementedError

    for dataset_name in dataset_names:
        print('Evaluating dataset: ', dataset_name)
        
        # load dataset 
        if dataset_name == 'uni-tsocial' \
            or dataset_name == 'mnist/dgl/mnist0' \
            or dataset_name == 'mnist/dgl/mnist1' \
            or dataset_name == 'mutag/dgl/mutag0' \
            or dataset_name == 'bm/dgl/bm_mn_dgl' \
            or dataset_name == 'bm/dgl/bm_ms_dgl' \
            or dataset_name == 'bm/dgl/bm_mt_dgl' \
            :
            dataset = Dataset(dataset_name, prefix='../datasets/unified/', sp_type=sp_type) #, debugnum=10000)
        elif dataset_name == 'reddit' \
            or dataset_name == 'weibo' \
            or dataset_name == 'weibo' \
            or dataset_name == 'amazon' \
            or dataset_name == 'yelp' \
            or dataset_name == 'tfinace' \
            or dataset_name == 'tolokers' \
            or dataset_name == 'questions' \
            or dataset_name == 'tfinance' \
            :
            dataset = Dataset(dataset_name+'-els', prefix='../datasets/edge_labels/', sp_type=sp_type, labels_have='ne')
        else:
            dataset = Dataset(dataset_name)

        results_df = pd.DataFrame()
        
        for kernel in kernels:
            print('Evaluating model: ', args.pretrain_model + '-' + kernel)
            for cross_mode in cross_modes:
                model_result_df = work(dataset, kernel, cross_mode, args)
                results_df = pd.concat([results_df, model_result_df], ignore_index=True)
        
        print(f"--- All results for dataset {dataset_name} ---")
        print(results_df)

if __name__ == "__main__":
    main()