import math
import torch
import torch.nn as nn

class Input_Embedding(nn.Module): 
    
    def __init__(self, vocab_size:int, d_model:int ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)
        
    def forward(self, x):
        # Section [3.4]
        # From paper: ... In the embedding layers, we multiply those weights by squareRoot(dmodel).
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class Positional_Encoding(nn.Module):
    
    def __init__(self, d_model:int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        # sequence_length: max length of sentence
        
        # matrix of shape (sequence_length, d_model)
        pos_enc = torch.zeros(sequence_length, d_model)
        
        # Section [3.5]
        #Postional Encoding:
        # pe(position, 2*index) = sin( position/10000^(2*index/d_model) )    -> @ even places
        # pe(position, 2*index+1) = cos( position/10000^(2*index/d_model) )  -> @ Odd places
        ## 
        
        # Vector for position
        # 1d
        positions = torch.arange(0, sequence_length, dtype=torch.float)
        
        # -> [1,2,3,4] -> [[1],[2],[3],[4]] 
        # #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        positions = positions.unsqueeze(1) # final shape: (sequence_length,1) -> 2D
        
        # calculate denominator in logspace for numerical stability
        #                     The 2*index part(let's say this is 'W')
        #                        ↙                             ↘
        denominator = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0) / d_model ))
        # Basically, log(10000 ^ (W / d_model))
        # => (W / d_model) * log(10000)
        # To 'cancel' out the log part, we use exponentiation
        # => exp((W / d_model) * log(10000))
        # Since this is in denominator, it will be:
        # => exp(-( w / d_model ) * ( log(10000 )))

        #Sine to even positions and Cosine to Odd positions
        #      (:)↙ All rows& ↙ here end is not given, meaning till last of array
        # array[:, start:end:step]
        pos_enc[:, 0::2] = torch.sin(positions * denominator)
        pos_enc[:, 1::2] = torch.cos(positions * denominator)
        
        # It was for one sentence,
        # For multiple sentences, we need to 'batch' them...
        # ..for a batch of sentences
        # [[1], [2], [3], [4]] -> [[[1], [2], [3], [4]]]
        self.pos_enc = pos_enc.unsqueeze(0) # dimension: (1, sequence_length, d_model) -> 3D
        
        
        # Save the positions to buffer..
        # ..so that it can be saved to the model's "state_dict"
        # A buffer is a tensor that is not considered a model parameter but is still part of the model's state. Buffers are 
        # typically used for tensors that need to be saved and loaded alongside the model but should not be updated by the 
        # optimizer during training. Examples include running statistics in batch normalization layers.
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer("postitons", pos_enc)
        
    def forward(self, x):
        # Current shape of self.positions: (batch, sequence_length, d_model) -> 3D
        # x.shape[1] -> length of the word or sequence
        x += (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False) # As these are fixed and will not change
        return self.dropout(x)


class Layer_Normalization(nn.Module):
    
    def __init__(self,features:int, eps:float = 10**-10)->None:
        super().__init__()
        self.eps = eps
        # nn.Parameter makes it learnable
        # y = ax + b
        self.alpha = nn.Parameter(torch.ones(features)) # multiply -> a
        self.beta = nn.Parameter(torch.zeros(features))  # add -> b
        
    def forward(self,x):
        mean = x.float().mean(dim=-1, keepdim = True)
        standard_deviation = x.float().std(dim=-1, keepdim = True)
        # X = (x - mean) / sqrt(standard deviation^2  + epsilon)
        # ignoring square root as value of epsilon is very very small
        return self.alpha * ((x-mean) / (standard_deviation + self.eps)) + self.beta
        
        
    
class Feed_Forward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float)-> None:
        super().__init__()
        # Section [3.3]
        # The dimensionality of input and output is dmodel=512, and the inner-layer has dimensionality df⁢f=2048.
        self.linear_layer_1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=True) # W1 and B1
        self.linear_layer_2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=True) # W2 and B2
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, d_model)
        # (batch, sequence_length, d_model) ---layer_1---->
        # (batch, sequence_length, d_ff) ----layer_2---->
        # (batch, sequence_length, d_model)
        return self.linear_layer_2(
            self.dropout(    # 2.Randomly sets some activation to zero to avoid overfitting
                torch.relu(  # 1. introduce non-linearity
                    self.linear_layer_1(x)
                )
            )
        )



class MultiHeadAttention(nn.Module):
    # Section [3.2.2]
    # d_v is same as d_k.
    # https://youtu.be/ISNdQcPhsts?si=nQdY7W3skefzGHs9&t=1776
    def __init__(self, d_model:int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0 , "d_model can't be divided into h parts equally"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # Wq
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # Wk
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # Wv
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)
        
        
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        
        # @ -> matrix multiplication
        # .transpose(-2, -1) -> swap last and second last dims
        # (batch,h, sequence_length, d_k) --> (batch,h, sequence_length, sequence_length)
        # (batch,h, sequence_length, d_k) @ (batch,h, d_k, sequence_length)
        # The operation (batch, h, sequence_length, d_k) @ (batch, h, d_k, sequence_length) performs a matrix multiplication along
        # the last two dimensions of the tensors.
        # Specifically, for each batch and each head, it multiplies a(sequence_length, d_k) matrix with a (d_k, sequence_length) 
        # matrix. -> output matrix dims are (sequence_length, sequence_length)
        # [Q * Kt / sqrt(d_k)]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -10e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, sequence_length, sequence_length)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # attention_scores @ value 
        # => (batch, h, sequence_length, sequence_length) @ (batch, h, sequence_length, d_k)
        # => (batch, h, sequence_length, d_k)
        return (attention_scores @ value), attention_scores # <--- used for vizualisation
        
    
    def forward(self, q,k,v, mask):
        # mask is used to hide words that we don't need, before doing the softmax part
        
        # query, key, value: (batch, sequence_length, d_model) --> (batch, sequence_length, d_model)
        query = self.w_q(q) # Q'
        key = self.w_k(k)   # K'
        value = self.w_v(v) # V'
        
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view
        # (batch, sequence_length, d_model) -view-> (batch, sequene_length, h, d_k) -transpose->
        # (batch, h, sequence_length, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1, 2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # x -> HEAD 1, HEAD 2, ... (from the slide)
        
        # x now is (batch, h, sequence_length, d_k) --transpose(1,2)-->
        # (batch, sequece_length, h, d_k) --view()-->
        # (batch, sequence_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # self.d_k = d_model//h
        
        # (batch, sequence_length, d_model) --> (batch, sequence_length, d_model)
        return self.w_o(x)
    
    
class Residual_Connection(nn.Module):
    # Connections that skips to Add&Norm
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm = Layer_Normalization(features)
        
    def forward(self, x, previous_layer):
        return x + self.dropout(previous_layer(self.norm(x)))
        
        
        
############################# Encoder ###################################

class Encoder_Block(nn.Module):
    
    def __init__(self, features: int,self_attention:MultiHeadAttention, feed_forward:Feed_Forward, dropout:float)->None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([Residual_Connection(features, dropout) for _ in range(2)])
        
    def forward(self,x, src_mask):
        # The lambda function is used to wrap the self_attention method call, allowing it to be passed as an argument to another 
        # function or method. This is useful in scenarios where the method being called requires additional parameters
        # (like src_mask), but the context in which it is being used only allows for a single argument (like x).
        # Basically used to capture 'src_mask' and pass it
        x = self.residual_connections[0](x, lambda x: self.self_attention(x,x,x, src_mask)) 
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.norm = Layer_Normalization(features)
        self.layers = layers
        
    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
    
########################### Decoder ########################################

class Decoder_Block(nn.Module):
    
    def __init__(self, features: int, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: Feed_Forward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.dropout=dropout
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([Residual_Connection(features, dropout) for _ in range(3)])
        
    # def forward(self, encoder_output, encoder_mask, x, decoder_mask):
    def forward(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        decoder_input = self.residual_connections[0](decoder_input, lambda decoder_input: self.self_attention(decoder_input,decoder_input,decoder_input,decoder_mask))
        # Query from Decoder, Key, Value from Encoder, Mask of the Encoder
        # => Encoder -> Key, Value, Mask; Decoder -> Query
        decoder_input = self.residual_connections[1](decoder_input, lambda x: self.cross_attention(decoder_input, encoder_output, encoder_output, encoder_mask))
        decoder_input = self.residual_connections[2](decoder_input, self.feed_forward)
        return decoder_input
    
class Decoder(nn.Module):
    
    def __init__(self, features: int,  layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Layer_Normalization(features)
        
    def forward(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        for layer in self.layers:
            decoder_input = layer(encoder_output, encoder_mask, decoder_input, decoder_mask)
        return self.norm(decoder_input)
    
    
    
########### Projection Layer(the last Linear layer above the decoder) ################
# Just like the output layer...basically it maps outputs to the classes(vocal)

class Projection_layer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
        
    def forward(self, x):
        # (batch, sequence_length, d_model) --> (batch, sequence_length, vocab_size)
        # return nn.LogSoftmax(self.projection(x), dim=-1) # Log Softmax for numercal stability
        return self.projection(x)
    
#######################################################################################
#######################################################################################

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: Input_Embedding, target_embedding: Input_Embedding, src_pos: Positional_Encoding, target_pos: Positional_Encoding, projection: Projection_layer)->None:
        super().__init__()
        self.encoder =encoder
        self.decoder= decoder
        self.src_embed = src_embedding
        self.target_embed = target_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection
        
    # 1.
    def encode(self, source, encoder_mask):
        s = self.src_embed(source)
        s = self.src_pos(s)
        return self.encoder(s, encoder_mask)
    
    # 2.
    def decode(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        t = self.target_embed(decoder_input)
        t = self.target_pos(t)
        return self.decoder(encoder_output, encoder_mask, t, decoder_mask)
    
    # 3. pass it to last project layer to 'merge' them to vocab 
    def project(self, x):
        return self.projection_layer(x)
    
##############################################################################################
##############################################################################################

def build_transformer(source_vocab_size: int, target_vocab_size:int, source_sequence_length:int, target_sequence_length:int, d_model:int=512, N_layers: int=6, h:int = 8, dropout:float=0.1, d_ff:int=2048):
    
    # 1. create embedding layer
    source_embeddings = Input_Embedding(vocab_size=source_vocab_size, d_model=d_model)
    target_embeddings = Input_Embedding(vocab_size=target_vocab_size, d_model=d_model)
    
    # 2. Pasitional Encoding Layer
    source_pos = Positional_Encoding(d_model, source_sequence_length, dropout)
    target_pos = Positional_Encoding(d_model, target_sequence_length, dropout)

    # 3. Encoder Block
    encoder_blocks = list()
    for _ in range(N_layers):
        encoder_self_attention_block=MultiHeadAttention(d_model=d_model,dropout=dropout,h=h)
        feed_forward_block=Feed_Forward(d_model=d_model, dropout=dropout, d_ff=d_ff)
        encoder_block=Encoder_Block(features=d_model,self_attention=encoder_self_attention_block, dropout=dropout,feed_forward=feed_forward_block)
        encoder_blocks.append(encoder_block)
        
    # 4. Decoder blocks
    decoder_blocks = list()
    for _ in range(N_layers):
        decoder_self_attention_block=MultiHeadAttention(d_model=d_model,dropout=dropout,h=h)
        decoder_cross_attention_block=MultiHeadAttention(d_model=d_model,dropout=dropout,h=h)
        feed_forward_block=Feed_Forward(dropout=dropout,d_ff=d_ff,d_model=d_model)
        decoder_block=Decoder_Block(features=d_model, cross_attention=decoder_cross_attention_block, self_attention=decoder_self_attention_block, feed_forward=feed_forward_block, dropout=dropout)
        decoder_blocks.append(decoder_block)
    
    # 5. Create Encoder and Decoder
    encoder = Encoder(features=d_model, layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(features=d_model, layers=nn.ModuleList(decoder_blocks))
    
    # 6. Prjection layerr
    projection_layer = Projection_layer(d_model=d_model,vocab_size=target_vocab_size)
    
    #7. Transformer
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embedding=source_embeddings,
                              target_embedding=target_embeddings,
                              src_pos=source_pos,
                              target_pos=target_pos,
                              projection=projection_layer)
    
    
    # 8. Initialize params
    for p in transformer.parameters():
        # Biases matrices are ofteh are one-dimensional, 
        # whereas weights matrices are two-dimensional
        # So, to avoid initializing baises, we check dims
        if p.dim()>1:
            # initialization method helps in maintaining the variance of the weights across layers,
            # which can lead to better training performance.
            nn.init.xavier_uniform_(p)
            
    return transformer
    
