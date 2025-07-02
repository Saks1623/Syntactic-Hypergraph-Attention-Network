import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn import metrics
from layers import *
from tqdm import tqdm

def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable

def trans_to_cpu(variable):
    return variable.cpu() if torch.cuda.is_available() else variable

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, 
                 dropout=self.dropout, alpha=0.1, transfer=False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size,
                 dropout=self.dropout, alpha=0.2, transfer=True, concat=False)
        
    def forward(self, x, H):
        # Numerical stability checks
        if torch.isnan(x).any():
            x = torch.nan_to_num(x)
        if torch.isnan(H).any():
            H = torch.nan_to_num(H)
            
        x = self.gat1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gat2(x, H)

class DocumentGraph(Module):
    def __init__(self, opt, pre_trained_weight, class_weights, n_node, n_categories, vocab_dic=None):
        super(DocumentGraph, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hiddenSize
        self.initial_feature = opt.initialFeatureSize
        
        # Initialize embedding layer
        if pre_trained_weight is not None:
            pre_trained_weight = torch.FloatTensor(pre_trained_weight)
            self.embedding = nn.Embedding.from_pretrained(
                pre_trained_weight, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(
                n_node + 1, self.initial_feature, padding_idx=0)
            nn.init.xavier_uniform_(self.embedding.weight)
        
        # Projection layer with better initialization
        self.projection = nn.Linear(self.initial_feature, self.hidden_size)
        nn.init.kaiming_normal_(self.projection.weight)
        
        # Enhanced attention mechanism
        self.syntax_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.syntax_gate = nn.Sigmoid()
        
        # HGNN with gradient clipping
        self.hgnn = HGNN_ATT(
            self.initial_feature, 
            self.initial_feature,
            self.hidden_size, 
            dropout=opt.dropout
        )
        
        # Prediction layers
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.prediction_transform = nn.Linear(
            self.hidden_size, n_categories, bias=True)
        
        if opt.normalization:
            self.layer_normC = nn.LayerNorm(n_categories, eps=1e-6)
        
        # Initialize loss function with class weights
        if class_weights is not None:
            if len(class_weights) < n_categories:
                print(f"Padding class weights from {len(class_weights)} to {n_categories}")
                class_weights = np.pad(class_weights, 
                                    (0, n_categories - len(class_weights)),
                                    'constant', 
                                    constant_values=1.0)
            
            print(f"Final class weights shape: {len(class_weights)} (should match {n_categories})")
            self.loss_function = nn.CrossEntropyLoss(
                weight=trans_to_cuda(torch.Tensor(class_weights).float()))
        else:
            self.loss_function = nn.CrossEntropyLoss()
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.opt.lr_dc_step,
            gamma=self.opt.lr_dc
        )

    def compute_scores(self, inputs, node_masks):
        # More stable masking and pooling
        node_masks = node_masks.unsqueeze(-1).float()
        hidden = inputs * node_masks
        sum_mask = torch.sum(node_masks, -2).clamp(min=1e-7)
        b = torch.sum(hidden, -2) / sum_mask
        b = self.layer_normH(b)
        return self.prediction_transform(b)

    def forward(self, inputs, HT):
        # Convert inputs to tensor if they aren't already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.LongTensor(inputs)
        inputs = inputs.to(self.embedding.weight.device)
        
        # Input validation
        inputs = inputs.clamp(0, self.embedding.num_embeddings-1)
        
        hidden = self.embedding(inputs)
        projected_hidden = self.projection(hidden)
        
        # Stabilize HGNN output
        semantic_nodes = self.hgnn(hidden, HT)
        if torch.isnan(semantic_nodes).any():
            semantic_nodes = torch.nan_to_num(semantic_nodes)
        
        combined = torch.cat([projected_hidden, semantic_nodes], dim=-1)
        syntax_weights = self.syntax_attention(combined)
        syntax_gate = self.syntax_gate(syntax_weights)
        
        return syntax_gate * semantic_nodes + (1 - syntax_gate) * projected_hidden

def forward(model, alias_inputs, HT, items, targets, node_masks):
    # Convert all inputs to tensors first
    items = torch.LongTensor(items)
    alias_inputs = torch.LongTensor(alias_inputs)
    HT = torch.stack(HT)
    node_masks = torch.FloatTensor(node_masks)
    
    # Now apply clamping to tensors
    items = items.clamp(0, len(model.embedding.weight)-1)
    alias_inputs = alias_inputs.clamp(0, items.shape[1]-1)
    
    # Move to GPU if available
    items = trans_to_cuda(items)
    alias_inputs = trans_to_cuda(alias_inputs)
    HT = trans_to_cuda(HT)
    node_masks = trans_to_cuda(node_masks)
    
    # Forward pass with anomaly detection
    with torch.autograd.detect_anomaly():
        node = model(items, HT)
        seq_hidden = torch.stack([node[i][alias_inputs[i]] 
            for i in torch.arange(len(alias_inputs)).long()])
    
    targets = trans_to_cuda(torch.LongTensor(targets))
    return targets, model.compute_scores(seq_hidden, node_masks)

def train_model(model, train_data, opt):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    
    slices = train_data.generate_batch(opt.batchSize, True)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = train_data.get_slice(i)
        
        model.optimizer.zero_grad()
        targets, scores = forward(model, alias_inputs, HT, items, targets, node_masks)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.4f' % (total_loss / len(slices)))

def test_model(model, test_data, opt, verbose=True):
    model.eval()
    test_pred, test_labels = [], []
    
    with torch.no_grad():
        slices = test_data.generate_batch(opt.batchSize, False)
        for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
            i = slices[step]
            alias_inputs, HT, items, targets, node_masks = test_data.get_slice(i)
            targets, scores = forward(model, alias_inputs, HT, items, targets, node_masks)
            test_labels.extend(targets.cpu().numpy())
            test_pred.extend(scores.argmax(-1).cpu().numpy())

    if verbose:
        print("Classification Report:")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        print("Macro F1:", metrics.f1_score(test_labels, test_pred, average='macro'))
        print("Micro F1:", metrics.f1_score(test_labels, test_pred, average='micro'))
    
    return metrics.classification_report(test_labels, test_pred, digits=4), \
           metrics.accuracy_score(test_labels, test_pred)