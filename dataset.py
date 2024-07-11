from typing import Any
import torch
import torch.nn
from torch.utils.data import Dataset



def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) # every value above diagonal preserved, rest 0
    return mask==0 # But we needed the opposite, so we do this to get all 1s(or True) in lower traingular matrix

class Hi_en_Dataset(Dataset):
    
    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_language, target_language, sequence_length):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_language= source_language
        self.target_language = target_language
        self.sequence_length = sequence_length
        self.sos_token = torch.tensor([tokenizer_source.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_source.token_to_id('[EOS]')], dtype=torch.int64)
        self.padding_token = torch.tensor([tokenizer_source.token_to_id('[PAD]')], dtype=torch.int64)
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Any:
        source_target_pair = self.dataset[index]
        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]
        
        # Text -> toekn -> input id
        encoder_input_tokens = self.tokenizer_source.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids
        
        # Add padding
        encode_num_padding_tokens = self.sequence_length - len(encoder_input_tokens) - 2 # -2, cuz we will be adding SOS & EOS too
        decode_num_padding_token = self.sequence_length - len(decoder_input_tokens) - 1 # we only add SOS, cuz we dont want the decoder to know the end of a sequence
        
        if encode_num_padding_tokens  < 0 or decode_num_padding_token < 0:
            raise ValueError("Sentence is too long")
        
        if self.sequence_length < len(encoder_input_tokens):
            encoder_input_tokens = encoder_input_tokens[:self.sequence_length-2]
        if self.sequence_length < len(decoder_input_tokens):
            decoder_input_tokens = decoder_input_tokens[:self.sequence_length-1]
        
        
        # Add SOS & PAD tokens and the required number of PAD tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.padding_token] * encode_num_padding_tokens, dtype = torch.int64)
            ],
            dim=0,
        )
        
        #Add only SOS & PAD tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.padding_token] * decode_num_padding_token, dtype=torch.int64)
            ],
            dim=0,
        )
            
        #  label(or class): the output from the decoder
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.padding_token] * decode_num_padding_token, dtype=torch.int64)
            ],
            dim=0,
        )
        
        
        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length
        
        return {
            "encoder_input" : encoder_input, # shape (sequence_length,)
            "decoder_input": decoder_input,  # shape (sequence_length,)
            "encoder_mask" : (encoder_input != self.padding_token).unsqueeze(0).unsqueeze(0).int(), # (1(for batch dim),1(for seq dim),Sequence_length)
            "decoder_mask": (decoder_input != self.padding_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # mask of size seq ken * seq len
            # => (1, sequence_length) & (1, sequence_length, sequence_length)
            "label" : label,
            "source_text" : source_text,
            "target_text" : target_text
            
        }
        
    
        
        
