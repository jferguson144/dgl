import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import EdgeSoftmax

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)



    def propagate_attn(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}
              
        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class RGCN_Attn_BlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, num_heads=1, 
                 bias=None, activation=None, self_loop=False, dropout=0.0, 
                 concat_attn=True, relation_type="block"):
        super(RGCN_Attn_BlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        # Attn stuff
        # attn head transformation to ensure that output vectors are same size as input vcetors

        self.relation_type = relation_type

        self.concat_attn = concat_attn
        self.num_heads = num_heads
        self.softmax = EdgeSoftmax()
        self.attn_k = nn.Parameter(torch.Tensor(size=(num_heads, in_feat, out_feat)))
        nn.init.xavier_uniform_(self.attn_k, gain=nn.init.calculate_gain('relu'))
        self.attn_q = nn.Parameter(torch.Tensor(size=(num_heads, out_feat, out_feat)))
        nn.init.xavier_uniform_(self.attn_q, gain=nn.init.calculate_gain('relu'))

        if concat_attn:
          attn_feat = int(out_feat / num_heads)
        else:
          attn_feat = out_feat
        self.attn_transforms = []
        for head in range(self.num_heads):
          self.attn_transforms.append(torch.nn.Linear(out_feat, attn_feat, bias=False))
          self.add_module("attn_transform_%d" %(head), self.attn_transforms[head])


        # assuming in_feat and out_feat are both divisible by num_bases
        if relation_type == "block":
          self.submat_in = in_feat // self.num_bases
          self.submat_out = out_feat // self.num_bases

          self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
          nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        elif relation_type == "vector":
          self.weight = nn.Parameter(torch.Tensor(self.num_rels, out_feat))
          nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
          

    def msg_func(self, edges):
        # multiply edge_values by edge_attn
        # expanded_edges has shape E x H x h
        expanded_edges = edges.data["edge_value"].unsqueeze(1).expand([-1, self.num_heads, -1])
        # intermediate layers: E x H x h/H. Final layer: E x H x h
        transformed_edges = []
        for head in range(self.num_heads):
          transformed_edges.append(self.attn_transforms[head](expanded_edges[:, head]))
        msg = torch.stack(transformed_edges, dim=1) * edges.data["unnormalized_attn"] / edges.dst["z"]
        if self.concat_attn:
          # Intermediate layer, concatenate each attention head
          final_msg = msg.view([msg.shape[0], -1])
        else:
          # Final layer, average each attention head
          final_msg = msg.sum(dim=1) / self.num_heads
        return {'msg': final_msg}

        
    def EdgeTransformBlock(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        edge_value = torch.bmm(node, weight).view(-1, self.out_feat)
        # Basically, we want to take this and scale it by an attention weight, removing the node normalization
        # Attention weight might have to be computed first in order to get the appropriate normalization for each edge
        
        return {'edge_value': edge_value}

    def EdgeTransformVector(self, edges):
      weight = self.weight.index_select(0, edges.data["type"])
      node = edges.src["h"]
      edge_value = node
      if weight.shape[1] < node.shape[1]:
        edge_value[:, :weight.shape[1]] += weight
      else:
        edge_value += weight
      return {"edge_value": edge_value}

    def EdgeAttention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and transformed dst
        # a is ExH
        a = (edges.dst['a1'] * edges.data['a2']).sum(-1, keepdim=True)

        # a is the edge transformation without exponentiation
        return {'a' : a}

    def EdgeSoftmax(self, g):
        scores, normalizer = self.softmax(g.edata['a'], g)
        # Save normalizer
        g.ndata['z'] = normalizer
        # Dropout attention scores and save them
        # jferguson: Removed attn dropout
        g.edata['unnormalized_attn'] = scores # self.attn_drop(scores)


    def propagate(self, g):
        # Apply attention to msg
        h = g.ndata["h"]
        if self.relation_type == "block":
          g.apply_edges(self.EdgeTransformBlock)
        elif self.relation_type == "vector":
          g.apply_edges(self.EdgeTransformVector)
        elif self.relation_type == "basis":
          pass
        # edges all have edge_value according to W_r * h_s
        # We now want to compute attention between h_d and edge_value
        # h has shape NxX. Want it to have shape NxHxD
        #ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        #head_ft = ft.transpose(0, 1)  # HxNxD'
        expanded_h = h.unsqueeze(0).expand([self.num_heads, -1, -1])
        e = g.edata["edge_value"]
        expanded_edges = e.unsqueeze(0).expand([self.num_heads, -1, -1])
        a1 = torch.bmm(expanded_h, self.attn_k).transpose(0, 1)  # NxHxh/H
        a2 = torch.bmm(expanded_edges, self.attn_q).transpose(0, 1)  # ExHxh/H

        #g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        g.ndata.update({'a1' : a1})
        g.edata.update({'a2' : a2})
        
        g.apply_edges(self.EdgeAttention)
        self.EdgeSoftmax(g)

        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)


    def apply_func(self, nodes):
        return {'h': nodes.data['h']}

