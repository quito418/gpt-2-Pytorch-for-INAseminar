'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        '''
        What are Query, Key, and Value?
        See https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
        
        How Attention was made?
        https://www.youtube.com/watch?v=XfpMkf4rD6E
        See this for history of RNN+attention http://cs231n.stanford.edu/slides/2022/lecture_11_ruohan.pdf
        '''
        w = torch.matmul(q, k.transpose(-2,-1))
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        '''
        Divide to multi-head Q, K, V
        Transpose dimension of 2:head and 1:time/token
        '''
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        '''
        Key, Value Cache logic
        Masked attention prevents tokens from attending to tokens that come after it in the sequence. 
        Hence, the value does not change. No need to recompute attention for tokens that have already been processed.
        Nevertheless, we save only the key and value, instead of the computed attention scores.
        '''
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key, value))  # transpose to have same shapes for stacking


        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        '''
        Translates integer ID to vector representation
        '''
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        '''
        Learned positional embeddings
        https://voidism.github.io/notes/2020/01/26/What-has-the-positional-embedding-learned/
        '''
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        
        '''
        GPT-2 can generate text incrementally, so it keeps track of "past" states 
        to maintain context without reprocessing the entire sequence each time. 
        Saves Query and Key of attention for reuse.
        '''
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        '''
        Positional encodings provide information about the order of tokens, 
        which is essential since the model's layers (being based on self-attention) 
        do not inherently process tokens in order. 
        (see page 67 of slite from http://cs231n.stanford.edu/slides/2022/lecture_11_ruohan.pdf)

        Generating position_ids dynamically allows the model to handle 
        variable-length sequences and continue generating text from where it left off.
        '''
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        '''
        Word integer IDs are given as input.
        Reshaping these IDs prepares them for processing through 
        the embedding layers and aligns them with the 
        expected input format of these layers.
        '''
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        
        '''
        Tokenization and positional encoding
        Why cosine, sin functions for positional encoding?
        See https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        TLDR: making position encoded into feature space that has distances between two vectors to better represent the distance between two token
        e.g., using binary encoding for positional encoding would result in two consecutive tokens having the very distant distance.
        Two example of sequential number in binary having significantly diffferent distance:
        011111111 -> 100000000  , 100000000 -> 100000001
        '''
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        '''
        Token type embeddings (if used) allow the model to distinguish between different segments 
        within the input (e.g., in tasks involving multiple sequences like question-answering). 
        https://huggingface.co/docs/transformers/glossary#token-type-ids
        '''
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        '''
        Why addition of position embedding, not concatenation?
        see  https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/?utm_source=reddit&utm_medium=web2x&context=3
        TLDR: in a high-dimensional space simply adding PE vector to input can be regarded as having a separate PE dimension
        '''
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents