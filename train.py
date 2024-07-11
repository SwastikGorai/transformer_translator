import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
from datasets import load_dataset
from dataset import Hi_en_Dataset, casual_mask
from model import build_transformer
from config import get_config, get_weights_file_path
from tqdm import tqdm

def get_all_sentences(dataset, language):
    for i in dataset:
        yield i['translation'][language]

def build_tokenizer(config, dataset, language):
    tokenizer_path  = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer= Tokenizer(WordLevel(unk_token="[UNK]")) # If tokenizer sees some token outside its vocab, it will replace it with "[UNK]"
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    ds_train_raw = load_dataset("cfilt/iitb-english-hindi", split="train")
    ds_validation_raw = load_dataset("cfilt/iitb-english-hindi", split="validation")
    ds_test_raw = load_dataset("cfilt/iitb-english-hindi", split="test")
    
    # Build tokenizer
    tokenizer_source = build_tokenizer(config, ds_train_raw, config['source_language'])
    tokenizer_target = build_tokenizer(config, ds_train_raw, config['target_language'])
    
    
    train = Hi_en_Dataset(dataset=ds_train_raw,
                          tokenizer_source=tokenizer_source,
                          tokenizer_target=tokenizer_target,
                          source_language=config['source_language'],
                          target_language=config['target_language'],
                          sequence_length=config['sequence_length']) 
    validation = Hi_en_Dataset(dataset=ds_validation_raw,
                          tokenizer_source=tokenizer_source,
                          tokenizer_target=tokenizer_target,
                          source_language=config['source_language'],
                          target_language=config['target_language'],
                          sequence_length=config['sequence_length'])
    test = Hi_en_Dataset(dataset=ds_test_raw,
                    tokenizer_source=tokenizer_source,
                    tokenizer_target=tokenizer_target,
                    source_language=config['source_language'],
                    target_language=config['target_language'],
                    sequence_length=config['sequence_length'])
    
    
    max_len_source, max_len_target = 0,0
    
    for i in ds_train_raw:
        source_ids = tokenizer_source.encode(i['translation'][config['source_language']]).ids
        target_ids = tokenizer_source.encode(i['translation'][config['source_language']]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))
        
    print(f"Max len of source sentence: {max_len_source}")
    print(f"Max len of target sentences: {max_len_target}")
    
     

    train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=1, shuffle=True) # batch size=1: to process each sentence one by one
    test_dataloader = DataLoader(test, batch_size=config['batch_size'], shuffle=True) 
    
    return train_dataloader, validation_dataloader, test_dataloader, tokenizer_source, tokenizer_target




def get_model(config, vocab_source_len, vocab_target_len ):
    model = build_transformer(source_vocab_size=vocab_source_len,
                              target_vocab_size=vocab_target_len,
                              source_sequence_length=config['sequence_length'],
                              target_sequence_length=config['sequence_length'],
                              d_model=config['d_model'])
    return model

    
    
    


def train_model(config):
    # Device cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, validation_dataloader, test_dataloader, tokenizer_source, tokenizer_target = get_dataset(config=config)
    model = get_model(config=config, vocab_source_len=tokenizer_source.get_vocab_size(), vocab_target_len=tokenizer_target.get_vocab_size()).to(device=device)
    
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    
    initial_epoch=0
    global_step=0
    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device=device)
    
    
    #Train loop
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator= tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # (batch, sequence_length)
            decoder_input = batch['decoder_input'].to(device) # (batch, sequence_length)
            encoder_mask = batch['encoder_mask'].to(device)   # (batch, 1, 1, sequence_length)
            decoder_mask = batch['decoder_mask'].to(device)   # (bathc, 1, sequence_length, sequence_length)
            
            
            # run them trhough the transformer
            encoder_output=model.encode(encoder_input, encoder_mask) # (batch, sequence_length, d_model)
            decoder_output = model.decoder(encoder_output,encoder_mask, decoder_input, decoder_mask) # (batch, sequence_length, d_model)
            
            projection_output = model.project(decoder_output) # (batch , sequence_length, target_vocab_size)
            
            #Compare 
            label = batch['label'].to(device=device) # (batch, sequence_length)
            
            #since we want to compare the output of the projection with the label
            # => (batch , sequence_length, target_vocab_size) --transpose--> (batch * sequence_length, target_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            
            
            batch_iterator.set_postfix({f"loss: {loss.item():6.3f}"})
            
            # Tensorboard
            writer.add_scalar('train loss', loss.item(), global_step=global_step)
            writer.flush()
            
            
            # backpropagate
            loss.backward()
            
            #update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step+=1
            
            
    # Save model at every epoch
    model_filename = get_weights_file_path(config=config, epoch=f"{epoch:02d}")
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'global_step' : global_step
        
    }, model_filename)
    
    
import warnings
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config=config)
            
            
            
            
            
            


