# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from timeit import default_timer as timer
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab
import gensim
from gensim.models import Word2Vec
from tokenize import tokenize, untokenize
import io
import re
#from nltk.stem import PorterStemmer

# Setting the device for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Examining the dataset
with open('english_python_data.txt',"r") as data_file:
  print(data_file.readlines()[:11]) # Printing out the last 5 lines of the data
  
#custom tokenizer for python code
def tokenize_python_code(python_code_str):
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(python_tokens)):
        tokenized_output.append(python_tokens[i].string)
    return tokenized_output

tokenizer = get_tokenizer('spacy',language='en_core_web_sm') #tokenizer function for the english text

# Creating a Regular Expression (Regex) pattern of urls to remove them
url_pattern = re.compile(r"https?://\S+")

src_doc = []
tgt_doc = []

i = 0
# Making a dataset
with open('english_python_data.txt',"r") as data_file:
  data_lines = data_file.readlines()
  dps = [] # List of dictionaries
  dp = None # The current problem and solution
  for line in data_lines:
    if line[0] == "#":
      tab_count = 0
      loop_indent = 0
      dict = {0: 0}
      if dp:
        dp['solution'] = ''.join(dp['solution'])
        dp['solution'] = re.sub(r'\n+', '\n', dp['solution']) #replaces multiple newlines with a single newline
        dp['solution'] = re.sub(r'(\n\t*)*$', '', dp['solution']) #removes any newlines at the end of the solution
        dp['solution'] = re.sub(r'^\n', '', dp['solution']) #removes any newlines at the beginning of the solution
          
        tgt_doc.append(tokenize_python_code(dp['solution']))  
        dps.append(dp)
      dp = {"question": None, "solution": []}
      dp['question'] = line[1:].strip("\n ") # Removing any \n in the question
      dp['question'] = re.sub(r'^\d+ ', "", dp['question']) # If the question starts with numbers, I remove them.
      dp['question'] = url_pattern.sub('',dp['question']) # Replacing any urls
      dp['question'] = dp['question'].lower() # lowercasing the question
      dp['question'] = re.sub(r"([.!?])","",dp['question']) # removing any punctuation
      src_doc.append(tokenizer(dp['question'])) # Splitting the question into words
    else:
      line = re.sub(r'( |\t)*\n( |\t)*', '\n', line) #replaces spaces before a newline with a  just a newline
      if line == '\n':
        continue
      
      space_count = len(line) - len(line.lstrip(' '))
      if loop_indent > space_count:
        tab_count = dict[space_count]
        loop_indent = space_count
      if(len(dp['solution'])!=0):
        if (bool(re.search(r':\n$', dp['solution'][-1]))):
          dict[space_count] = tab_count
          loop_indent = space_count

      
      if tab_count > 0 and bool(re.search(r',\n', dp['solution'][-1])!=True) and bool(re.search(fr'^(\t){{{tab_count}}}', line))!=True:
        line = re.sub(r'^ *', '\t'*tab_count, line)#replaces the first space with a tab
      
      if re.search(r'^(if|else|elif|for|while|def|class|try|except|finally|with)', line):
        tab_count += 1 
      
      
      dp['solution'].append(line)
      

# converting the data to a table for easier viewing
dataset = pd.DataFrame(dps)

src_emb = gensim.models.Word2Vec.load("src_emb.model") # Loading the source language model
tgt_emb = gensim.models.Word2Vec.load("tgt_emb.model") # Loading the target language model

# Creating dictionaries for the tokenizers and the vocabularies
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'python'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3 # Tokens for Unknown, Padding, start of sentence, end of sentence
vocabularies = {}

vocabularies[SRC_LANGUAGE] = vocab(src_emb.wv.key_to_index, min_freq=0) # Creating a vocabulary for the source language
vocabularies[TGT_LANGUAGE] = vocab(tgt_emb.wv.key_to_index, min_freq=0) # Creating a vocabulary for the target language
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocabularies[ln].set_default_index(UNK_IDX)
  
# Positional Encoding module -> this class is the positional encoder (see above for details)
class PositionalEncoding(nn.Module):
  def __init__(self,emb_size:int, dropout:float, maxlen:int = 5000):
    super().__init__()
    den = torch.exp(-torch.arange(0,emb_size,2)*math.log(10000) / emb_size)
    pos = torch.arange(0,maxlen).reshape(maxlen,1)
    pos_embedding = torch.zeros((maxlen,emb_size))
    pos_embedding[:,0::2] = torch.sin(pos * den)
    pos_embedding[:,1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)

    # Saving the positional encoding in the model state dict, but making sure PyTorch doesn't "train"
    # these parameters because they don't need to be trained
    self.register_buffer('pos_embedding',pos_embedding)

  def forward(self,token_embedding: Tensor):
    return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# # Converting the tokens into embeddings
# class TokenEmbedding(nn.Module):
#   def __init__(self,vocab_size: int, emb_size):
#     super().__init__()
#     self.embedding = nn.Embedding(vocab_size,emb_size)
#     self.embed_size = emb_size

#   def forward(self,tokens:Tensor):
#     return self.embedding(tokens.long()) * math.sqrt(self.embed_size) # we multiply by square root of embedding size to scale. The Transformer paper mentions this.

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, emb_size: int, word2vec_model_path: str):
    super().__init__()
    self.word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
    self.embedding = nn.Embedding(vocab_size+4, emb_size)
    self.embed_size = emb_size

    # Initialize the embedding weights with the Word2Vec vectors
    self.embedding.weight.data[4:].copy_(torch.from_numpy(self.word2vec_model.wv.vectors))

  def forward(self, tokens: Tensor):
    return self.embedding(tokens.long()) * math.sqrt(self.embed_size)
  
    

# The Actual Model
class Seq2SeqTransformer(nn.Module):
  def __init__(self, num_encoder_layers:int, num_decoder_layers:int, emb_size:int, nhead:int, src_vocab_size:int, tgt_vocab_size:int, dim_feedforward: int=512, dropout:float = 0.1):
    super().__init__()
    self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,dropout=dropout,
                                      batch_first=True)
    self.generator = nn.Linear(emb_size,tgt_vocab_size) # A layer to convert the matrix (seq_len, emb_size) to (seq_len, tgt_vocab_size)
    self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size, "src_emb.model")
    self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size,emb_size, "tgt_emb.model")

    # Getting the positional encodings
    self.positional_encoding = PositionalEncoding(emb_size,dropout=dropout)

  def forward(self, src:Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor,
              memory_key_padding_mask: Tensor):

    # Embedding both the input and output
    src_embedding = self.positional_encoding(self.src_tok_emb(src))
    tgt_embedding = self.positional_encoding(self.tgt_tok_emb(trg))

    # Getting the output
    output = self.transformer(src_embedding, tgt_embedding, src_mask, tgt_mask, None, src_padding_mask,tgt_padding_mask,memory_key_padding_mask)

    # Getting the logits
    return self.generator(output)
  

  # Encoding the input
  def encode(self, src: Tensor, src_mask: Tensor):
    embedding = self.positional_encoding(self.src_tok_emb(src))
    encoder_output = self.transformer.encoder(embedding, src_mask)
    return encoder_output

  # Decoding the output
  def decode(self,tgt:Tensor, memory: Tensor, tgt_mask:Tensor):
    tgt_embedding = self.tgt_tok_emb(tgt)
    return self.transformer.decoder(self.positional_encoding(tgt_embedding), memory, tgt_mask)

# Defining the lookahead mask that will prevent the model from looking ahead during training
# Also need to define masks that will mask the padding tokens.
# If we don't mask the padding tokens, the model will end up taking the values of the padding into account
# into prediction

# Generating the lookahead mask
def generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones((sz,sz),device=DEVICE)) == 1).transpose(0,1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

# Creating the other mask
def create_mask(src, tgt):
  src_seq_len = src.shape[1]
  tgt_seq_len = tgt.shape[1]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

  src_padding_mask = (src == PAD_IDX)
  tgt_padding_mask = (tgt == PAD_IDX)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask     

# Splitting the data into training and testing
#training, testing = train_test_split(dataset,test_size=0.2,random_state=42,shuffle=True)

train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
training, testing = random_split(dataset,[train_size,test_size])

# Running the data through a pipeline to get the transformed and prepared dataset
# helper function to club together sequential operations
def sequential_transforms(*transforms):
  def func(txt_input):
    for transform in transforms:
      txt_input = transform(txt_input)
    return txt_input
  return func

# Function to add BOS/EOS and create tensor for input sequence indicies
def tensor_transform(token_ids):
  return torch.cat((torch.tensor([SOS_IDX]),torch.tensor(token_ids),torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  text_transform[ln] = sequential_transforms(tokenizer,vocabularies[ln],tensor_transform) # Tokenize, Convert to Indicies, then Add Special Tokens

# function to put all the data samples into batches
def collate_fn(batch):
  src_batch, tgt_batch = [], []

  # Iterating through the questions
  for X in batch.dataset['question'].values:
    token_tensor = text_transform[SRC_LANGUAGE](X.strip('\n\t'))
    #token_tensor = token_tensor[:-1]
    src_batch.append(token_tensor)

  # Iterating through the solutions
  for y in batch.dataset['solution'].values:
    token_tensor = text_transform[TGT_LANGUAGE](y.strip('\n\t'))
    #token_tensor = token_tensor[:-1]
    tgt_batch.append(token_tensor)

  src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
  tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
  return src_batch.T, tgt_batch.T

# Defining the model, loss function, and optimizer
torch.manual_seed(10)

SRC_VOCAB_SIZE = len(vocabularies[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocabularies[TGT_LANGUAGE])
EMB_SIZE = 64
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# Defining the model
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# Setting the parameters using the xavier uniform distribution
for p in transformer.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p)

# Putting the model on GPU
transformer = transformer.to(DEVICE)

# Defining the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # makes sure that the padding token doesn't contribute to the loss function!

# Defining the optimizer
optimizer = optim.AdamW(transformer.parameters(),lr=0.0001)

def train_epoch(model,optimizer):
  # Setting the model to training mode
  model.train()
  losses = 0

  # Preparing the data
  X,y = collate_fn(training)
  training_dataset = TensorDataset(X,y)
  train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE)

  # Iterating through the data
  for src, tgt in train_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)
    print(tgt.shape)
    tgt_in = tgt[:,:-1]

    # Getting the masks
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_in)
    logits = model(src,tgt_in, src_mask, tgt_mask, src_padding_mask,tgt_padding_mask,src_padding_mask) # memory is the encoder outputs

    optimizer.zero_grad()
    #print(tgt)
    tgt_out = tgt[:,1:]
    tgt_out = nn.functional.one_hot(tgt_out, num_classes=TGT_VOCAB_SIZE).to(DEVICE)
    print(tgt_out)
    #print(tgt_out)
    #print(logits)
    #print(tgt_out.shape)
    #logits = logits.contiguous().view(-1, logits.shape[-1])
    #tgt_out = tgt_out.contiguous().view(-1)
    #loss = loss_fn(logits.reshape(logits.size(0),-1,6969),tgt_out)
    #print(logits.shape)
    #print(tgt_out.shape)
    loss = loss_fn(logits,tgt_out)
    loss.backward() # Back propagation, calculating the gradients

    optimizer.step()
    losses += loss.item()

  return losses / len(list(train_dataloader)) # Getting the average loss per example

# Evaluation Loop
def evaluate(model):
  model.eval()
  losses = 0

  # Preparing the data
  X,y = collate_fn(testing)
  testing_data = TensorDataset(X,y)
  val_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE)

  # Iterating through the data
  for src, tgt in val_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:,:-1] # Getting the sentence except the EOS since EOS is never inputted to decoder

    # Getting the masks
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    logits = model(src,tgt_input, src_mask, tgt_mask, src_padding_mask,tgt_padding_mask,src_padding_mask) # memory is the encoder outputs
    tgt_out = tgt[:,1:]
    loss = loss_fn(logits.reshape(-1,logits.shape[-1]),tgt_out.reshape(-1))
    losses += loss.item()

  return losses / len(list(val_dataloader)) # Getting the average loss per example

# Training the model
NUM_EPOCHS = 10

for epoch in range(1,NUM_EPOCHS+1):
  start_time = timer()
  train_loss = train_epoch(transformer, optimizer)
  end_time = timer()
  val_loss = evaluate(transformer)
  print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')
  print(f'Epoch time: {(end_time - start_time):.3f}s')