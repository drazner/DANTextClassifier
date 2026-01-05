import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    # Initialize embedding matrix with small random values for OOV tokens.
    scale = 0.08
    emb = np.random.uniform(-scale, scale, (len(vocab), emb_size)).astype(np.float32)

    # Keep the padding vector at zero when a pad token exists.
    if vocab.pad_id is not None:
        emb[vocab.pad_id] = np.zeros(emb_size, dtype=np.float32)

    # Read file
    with open(emb_file, "r", encoding="utf-8", errors="ignore") as f:
        # Read embeddings line by line and fill the matrix for known words.
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != emb_size + 1:
                continue
            word = parts[0]
            if word in vocab:
                emb[vocab.word2id[word]] = np.asarray(parts[1:], dtype=np.float32)

    # Return the embedding matrix as a numpy array.
    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        # Cache frequently used hyperparameters.
        nwords = len(self.vocab)
        emb_size = self.args.emb_size
        hid_size = self.args.hid_size
        hid_layer = self.args.hid_layer

        # Embedding layer with optional padding index.
        self.embedding = nn.Embedding(nwords, emb_size, padding_idx=self.vocab.pad_id)
        
        # Dropout settings used in forward for word/embedding/hidden layers.
        self.word_drop = self.args.word_drop
        self.emb_dropout = nn.Dropout(self.args.emb_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        # Pooling strategy and activation function for the DAN encoder.
        self.pooling_method = self.args.pooling_method
        self.activation = nn.ReLU()

        # Build a stack of hidden feedforward layers.
        self.ff_layers = nn.ModuleList()
        input_dim = emb_size
        for _ in range(hid_layer):
            self.ff_layers.append(nn.Linear(input_dim, hid_size))
            input_dim = hid_size

        # Final projection to tag space. Tag_size is number of distinct output labels/classes 
        self.out_layer = nn.Linear(input_dim, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        '''
        1. Use uniform sampling from a range [-v, v]
        2. Set v to a number (i.e. v = 0.08)
        3. Pass hyperparameters (explicitly or use self.args)
        '''

        # Use a small uniform range for stable training.
        v = 0.08
        # Initialize all learnable parameters uniformly.
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.uniform_(param, -v, v)
        # Keep the padding embedding at zero to avoid affecting pooled features.
        if self.vocab.pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.vocab.pad_id].fill_(0.0)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        '''
        1. Load pre-trained word embeddings from numpy.array to nn.embedding
        2. Call self.load_embeddings(vocab, emb_file, emb_size)? 
        3. Convert from numppy array to nn embedding tensor
        4. Set nn.embedding param
        '''

        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size) 
        emb_tensor = torch.from_numpy(emb) 
        with torch.no_grad(): 
            self.embedding.weight.copy_(emb_tensor)


    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        # Look up embeddings for each token id 
        emb = self.embedding(x) # [batch_size, seq_length, emb_size]

        # Build mask for valid (non-pad) tokens 
        if self.vocab.pad_id is None: 
            valid_mask = torch.ones_like(x, dtype=torch.bool)
        else: 
            valid_mask = x.ne(self.vocab.pad_id) 

        # Randomly drop some token embeddings during training (word dropout)
        if self.training and self.word_drop > 0.0: 
            keep_mask = torch.rand_like(x.float()).gt(self.word_drop) & valid_mask 
        else: 
            keep_mask = valid_mask 

        # Apply the keep mask to embeddings before pooling
        emb = emb * keep_mask.unsqueeze(-1).float() 
        emb = self.emb_dropout(emb) 

        # Pool token embeddings into a single sentence representation 
        if self.pooling_method == "sum": 
            pooled = emb.sum(dim=1) 
        elif self.pooling_method == "max": 
            neg_inf = torch.finfo(emb.dtype).min
            masked_emb = emb.masked_fill(~keep_mask.unsqueeze(-1), neg_inf)
            pooled = masked_emb.max(dim=1).values
        else: #"avg"
            denom = keep_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = emb.sum(dim=1) / denom 

        # Feedforward layers with activation and dropout 
        h = pooled 
        for layer in self.ff_layers: 
            h = self.activation(layer(h)) 
            h = self.hid_dropout(h) 

        # Linear classification layer produces unnormalized scores 
        scores = self.out_layer(h) 
        return scores