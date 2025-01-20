import math
import random
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import sys
from model.DataSet import MEGADDataSet
from model.evaluation import eva
FType = torch.FloatTensor
LType = torch.LongTensor
DID = 0

class MEGAD:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_MEGAD_%d.emb'
        self.feature_path = './pretrain/%s_feature.emb' % self.the_data
        self.label_path = '../data/%s/node2label.txt' % self.the_data
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.best_AUCROC = 0
        self.best_AUCPR = 0
        self.data = MEGADDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()
        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)
        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb])
        self.loss = torch.FloatTensor()
        self.weight=torch.ones(1, self.node_emb.shape[1], dtype=torch.float).cuda()
        self.epslion_update_att_node_cos=0.01
        self.att_node_cos=self.get_att_node_cos(self.node_emb)

    def read_label(self):
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def get_dimension_variance_loss(self,emb,weights):   
        variances = torch.var(emb, dim=0)
        loss = torch.sum( torch.exp(-variances) * weights)
        return loss

    def update_dimension_variance_weight(self,dimension_variance_weight):
        self.weight = torch.nn.functional.softmax(torch.add(self.weight,torch.div(dimension_variance_weight, torch.norm(dimension_variance_weight))), dim=1)

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        att_cos_node=self.att_node_cos.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  
        res_st_loss = torch.mean((1 - new_st_adj)**2 * att_cos_node)
        res_sh_loss = torch.mean((1 - new_sh_adj)**2 * att_cos_node.sum(dim=0, keepdims=False))
        res_sn_loss = torch.mean((0 - new_sn_adj)**2 * att_cos_node.sum(dim=0, keepdims=False))
        l_rec = res_st_loss + res_sh_loss + res_sn_loss
        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()
        p_lambda = p_mu + (att * p_alpha ).sum(dim=1)  # [b]
        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha).sum(dim=1)
        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
        loss_dimension_variance=self.get_dimension_variance_loss(s_node_emb,self.weight)
        eta=0.5
        total_loss = eta*(loss.sum() + l_rec)+(1-eta)*loss_dimension_variance
        return total_loss
 
    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def get_att_node_cos(self,emb,k=2):
        n, m = emb.size()
        indices = torch.randperm(n)[:k]
        sampled_emb = emb[indices]
        distances = torch.norm(emb.unsqueeze(1) - sampled_emb, dim=2)
        distances = 2 * (1 - torch.sigmoid(distances))
        att_node = torch.mean(distances, dim=1, keepdim=True)
        att_node = att_node.clone().detach().requires_grad_(True)
        return att_node

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=1)
            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()
                self.update(sample_batched['source_node'].type(LType).cuda(),
                            sample_batched['target_node'].type(LType).cuda(),
                            sample_batched['target_time'].type(FType).cuda(),
                            sample_batched['neg_nodes'].type(LType).cuda(),
                            sample_batched['history_nodes'].type(LType).cuda(),
                            sample_batched['history_times'].type(FType).cuda(),
                            sample_batched['history_masks'].type(FType).cuda())

            if epoch<2:
                AUCROC, AUCPR,dimension_variance_weight = eva(epoch, self.labels, self.node_emb)
                self.update_dimension_variance_weight(dimension_variance_weight)
            else:
                AUCROC, AUCPR= eva(epoch, self.labels, self.node_emb)
            if random.random()>self.epslion_update_att_node_cos:
                self.att_node_cos=self.get_att_node_cos(self.node_emb)

            if AUCROC> self.best_AUCROC:
                self.best_AUCROC= AUCROC
                self.best_AUCPR= AUCPR
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))
            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('AUCROC(%.4f) AUCPR(%.4f)\n' % (AUCROC,AUCPR))
            sys.stdout.flush()
        print('Best performance:  AUC(%.4f) AUCPR(%.4f)' %(self.best_AUCROC,self.best_AUCPR))
        

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()
