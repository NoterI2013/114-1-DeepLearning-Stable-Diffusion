import tensorflow as tf
from transformers import CLIPTokenizer, TFCLIPTextModel
import numpy as np
import re

class TextEncoderWrapper:
    def __init__(self, clip_version="openai/clip-vit-large-patch14"):
        self.clip_version = clip_version
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        # self.clip_text_encoder = TFCLIPTextModel.from_pretrained(clip_version, from_pt=True, use_safetensors=False)
        self.clip_text_encoder = TFCLIPTextModel.from_pretrained(clip_version, from_pt=True, use_safetensors=True)
        self.clip_text_encoder.trainable = False

    def id2Sent(self, id_list, id2word_dict):
        # This function was originally standalone but logically belongs near text processing
        # However, it depends on id2word_dict which is data-specific.
        # We can keep it static or pass the dict.
        words = []
        for idx in id_list:
            idx_str = str(int(idx))
            if idx_str in id2word_dict:
                word = id2word_dict[idx_str]
                if word != "<PAD>" and word!="<RARE>":
                    words.append(word)
        sentence = " ".join(words)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        return sentence

    def id2clip(self, ids, id2word_dict):
        sentence = self.id2Sent(ids, id2word_dict)
        token_dict = self.clip_tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="tf"
        )
        
        outputs = self.clip_text_encoder(
            input_ids=token_dict["input_ids"],
            attention_mask=token_dict["attention_mask"],
            training=False
        )
        seq_emb = outputs.last_hidden_state           
        return seq_emb 
