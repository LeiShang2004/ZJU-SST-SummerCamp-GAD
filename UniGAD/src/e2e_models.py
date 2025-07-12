from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import pprint
from tqdm import tqdm

from utils import *
from predictors import *
from Pareto_fn import pareto_fn
from pcgrad_fn import pcgrad_fn

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def get_rec_at_k(labels, probs, k):
    sorted_indices = np.argsort(probs)[::-1]
    top_k_indices = sorted_indices[:k]
    true_positives_at_k = np.sum(labels[top_k_indices])
    total_true_positives = np.sum(labels)
    if total_true_positives == 0:
        return 0.0
    recall_at_k = true_positives_at_k / total_true_positives
    return recall_at_k

LABEL_DICT_KEYS = {
    'n':"node_labels",
    'e':'edge_labels',
    'g':'graph_labels',
}

class UnifyMLPDetector(object):
    def __init__(self, pretrain_model, dataset, dataloaders, cross_mode, args):
        self.args = args
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

        input_route, output_route = cross_mode.split('2')
        self.input_route = [c for c in input_route]
        self.output_route = [c for c in output_route]

        self.model = UNIMLP_E2E(
            in_feats=pretrain_model.in_dim,
            embed_dims=pretrain_model.embed_dim,
            khop=args.khop,
            activation=args.act_ft,
            graph_batch_num=args.batch_size,
            stitch_mlp_layers=args.stitch_mlp_layers,
            final_mlp_layers=args.final_mlp_layers,
            pretrain_model=pretrain_model,
            output_route=output_route,
            input_route=input_route,
            dropout_rate=args.dropout
        ).to(args.device)

        self.loss_weight_dict = {}
        if 'n' in self.output_route and hasattr(dataset, 'node_label') and dataset.node_label:
            node_ab_count = sum([x.sum() for x in dataset.node_label])
            node_total_count = sum(x.shape[0] for x in dataset.node_label)
            if node_ab_count > 0 and node_total_count > node_ab_count:
                self.loss_weight_dict['n'] = ( (node_total_count - node_ab_count) / node_ab_count, args.node_loss_weight)
        
        if 'e' in self.output_route and hasattr(dataset, 'edge_label') and dataset.edge_label:
            edge_ab_count = sum([x.sum() for x in dataset.edge_label])
            edge_total_count = sum(x.shape[0] for x in dataset.edge_label)
            if edge_ab_count > 0 and edge_total_count > edge_ab_count:
                self.loss_weight_dict['e'] = ( (edge_total_count - edge_ab_count) / edge_ab_count, args.edge_loss_weight)

        if 'g' in self.output_route and hasattr(dataset, 'graph_label') and dataset.graph_label:
            graph_ab_count = dataset.graph_label.sum()
            graph_total_count = dataset.graph_label.shape[0]
            if graph_ab_count > 0 and graph_total_count > graph_ab_count:
                self.loss_weight_dict['g'] = ( (graph_total_count - graph_ab_count) / graph_ab_count, args.graph_loss_weight)

        if dataset.is_single_graph:
            self.model.mask_dicts = {
                'n': {'train': dataset.train_mask_node_cur, 'val': dataset.val_mask_node_cur, 'test': dataset.test_mask_node_cur},
                'e': {'train': dataset.train_mask_edge_cur, 'val': dataset.val_mask_edge_cur, 'test': dataset.test_mask_edge_cur}
            }
            self.model.single_graph = True

        self.best_score = -1
        self.patience_knt = 0

    def get_loss(self, logits_dict={}, labels_dict={}):
        loss_items_dict = {k: 0 for k in self.output_route}
        loss_list = []
        w_list = []
        c_list = []

        for o_r in logits_dict:
            if labels_dict.get(LABEL_DICT_KEYS[o_r]) is None: continue
            
            weight = torch.tensor([1., self.loss_weight_dict.get(o_r, (1.0,))[0]], device=self.args.device)
            partial_loss = F.cross_entropy(logits_dict[o_r], labels_dict[LABEL_DICT_KEYS[o_r]], weight=weight)
            
            if o_r in self.input_route:
                loss_list.append(partial_loss)
                w_list.append(1.0 / len(self.input_route))
                c_list.append(0.01)
            loss_items_dict[o_r] = partial_loss.item()
        
        if not loss_list: return torch.tensor(0.0, device=self.args.device, requires_grad=True), loss_items_dict

        new_w_list = pareto_fn(w_list, c_list, model=self.model, num_tasks=len(loss_list), loss_list=loss_list)
        loss = sum(new_w_list[i] * loss_list[i] for i in range(len(loss_list)))
        
        return loss, loss_items_dict

    @torch.no_grad()
    def get_probs(self, logits_dict={}):
        probs_dict = {}
        for o_r in logits_dict:
            probs_dict[o_r] = logits_dict[o_r].softmax(1)[:, 1]
        return probs_dict

    @torch.no_grad()
    def _single_eval(self, labels, probs):
        score = {}
        if labels is None or probs is None: return {}
        
        labels = labels.cpu().numpy()
        probs = probs.cpu().numpy()
        
        k = int(np.sum(labels))
        if k == 0: k = 1

        if np.any(labels) and len(np.unique(labels)) > 1:
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
        else:
            score['AUROC'] = 0.5
            score['AUPRC'] = 0.0

        score['MacroF1'] = get_best_f1(labels, probs)[0]
        score['Rec_at_K'] = get_rec_at_k(labels, probs, k)
        score['K_value'] = k
        return score
    
    @torch.no_grad()
    def eval(self, labels_dict, probs_dict):
        result = {}
        for k in self.output_route:
            result[k] = self._single_eval(labels_dict.get(k), probs_dict.get(k))
        return result

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_ft, weight_decay=self.args.l2_ft)
        final_score_test = {}

        for epoch in tqdm(range(self.args.epoch_ft)):
            self.model.train()
            loss_items_total_train = {k: 0 for k in self.output_route}
            for batched_data in self.train_dataloader:
                batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                batched_graph = batched_graph.to(self.args.device)
                for k,v in batched_labels_dict.items():
                    if v is not None: batched_labels_dict[k] = v.to(self.args.device)
                batched_khop_graph = batched_khop_graph.to(self.args.device)

                logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='train')
                loss, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)

                for k_loss in loss_items_total_train:
                    loss_items_total_train[k_loss] += loss_items.get(k_loss, 0)
                
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            with torch.no_grad():
                self.model.eval()
                labels_dict_val_mul = {k:[] for k in self.output_route }
                probs_dict_val_mul = {k:[] for k in self.output_route }
                loss_items_total_val = {k: 0 for k in self.output_route}

                for batched_data in self.val_dataloader:
                    batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                    batched_graph = batched_graph.to(self.args.device)
                    for k,v in batched_labels_dict.items():
                        if v is not None: 
                            batched_labels_dict[k] = v.to(self.args.device)
                            if k[0] in self.output_route: labels_dict_val_mul[k[0]].append(v)
                    batched_khop_graph = batched_khop_graph.to(self.args.device)
                    
                    logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='val')
                    _, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)
                    for k_loss in loss_items_total_val:
                        loss_items_total_val[k_loss] += loss_items.get(k_loss, 0)
                    
                    probs = self.get_probs(logits_dict)
                    for k_prob in probs:
                        probs_dict_val_mul[k_prob].append(probs[k_prob])

                score_val = self.eval({k: torch.cat(v) for k, v in labels_dict_val_mul.items() if v}, 
                                      {k: torch.cat(v) for k, v in probs_dict_val_mul.items() if v})
                
                score_overall_val = np.mean([v[self.args.metric] for k, v in score_val.items() if v])

                log_loss(['Train', 'Val'], [loss_items_total_train, loss_items_total_val])

                if score_overall_val > self.best_score or self.patience_knt >= self.args.patience:
                    self.best_score = score_overall_val
                    self.patience_knt = 0
                    
                    labels_dict_test_mul = {k:[] for k in self.output_route }
                    probs_dict_test_mul = {k:[] for k in self.output_route }
                    for batched_data in self.test_dataloader:
                        batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                        batched_graph = batched_graph.to(self.args.device)
                        for k,v in batched_labels_dict.items():
                            if v is not None:
                                batched_labels_dict[k] = v.to(self.args.device)
                                if k[0] in self.output_route: labels_dict_test_mul[k[0]].append(v)
                        batched_khop_graph = batched_khop_graph.to(self.args.device)

                        logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='test')
                        probs = self.get_probs(logits_dict)
                        for k_prob in probs:
                            probs_dict_test_mul[k_prob].append(probs[k_prob])

                    final_score_test = self.eval({k: torch.cat(v) for k, v in labels_dict_test_mul.items() if v}, 
                                                 {k: torch.cat(v) for k, v in probs_dict_test_mul.items() if v})
                    print(f'Epoch {epoch}: New best score on val -> {self.best_score:.4f}')
                    print(pprint.pformat(final_score_test))
                    if self.patience_knt >= self.args.patience:
                        print("Patience exceeded, stopping early.")
                        break
                else:
                    self.patience_knt += 1

        return final_score_test