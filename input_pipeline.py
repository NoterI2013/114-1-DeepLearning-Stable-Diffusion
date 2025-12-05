import tensorflow as tf
import numpy as np
import pandas as pd
import re
import string
import os

# --- Constants are good to keep here or pass in ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNEL = 3
RESIZE_HEIGHT = 156
RESIZE_WIDTH = 156

# --- Text Preprocessing ---

def sent2IdList(line, word2Id_dict, MAX_SEQ_LENGTH=20):
    MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
    padding = 0
    # data preprocessing, remove all puntuation in the texts
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('  ', ' ')
    prep_line = prep_line.replace('.', '')
    tokens = prep_line.split(' ')
    tokens = [
        tokens[i] for i in range(len(tokens))
        if tokens[i] != ' ' and tokens[i] != ''
    ]
    l = len(tokens)
    padding = MAX_SEQ_LIMIT - l
    
    # make sure length of each text is equal to MAX_SEQ_LENGTH, and replace the less common word with <RARE> token
    for i in range(padding):
        tokens.append('<PAD>')
    line = [
        word2Id_dict[tokens[k]]
        if tokens[k] in word2Id_dict else word2Id_dict['<RARE>']
        for k in range(len(tokens))
    ]

    return line

def id2Sent(id_list, id2word_dict):
    words = []
    for idx in id_list:
        idx_str = str(int(idx))   # id 可能是 int 或 numpy.int，需要轉成 str 才能 index dict
        if idx_str in id2word_dict:
            word = id2word_dict[idx_str]
            if word != "<PAD>" and word!="<RARE>":
                words.append(word)
    sentence = " ".join(words)
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence

# --- Image Processing ---

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # 轉為 [0, 1]
    return img

def process_train_data(seq_emb, image_path):
    img = load_image(image_path)
    
    # === Data Augmentation 開始 ===
    
    # A. 隨機縮放與裁切 (模擬 Zoom In/Shift)
    # 先放大到 72x72
    img = tf.image.resize(img, size=(RESIZE_HEIGHT, RESIZE_WIDTH))
    # 再隨機裁切回 64x64
    img = tf.image.random_crop(img, size=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    
    # B. 隨機左右翻轉 (對花朵資料集非常有效)
    img = tf.image.random_flip_left_right(img)
    
    # C. 隨機亮度與對比度 (增加光影變化的魯棒性)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # D. 確保數值仍在 [0, 1] 範圍內 (因為亮度調整可能使其超出範圍)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    # === Data Augmentation 結束 ===
    
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    return seq_emb, img

def process_val_data(seq_emb, image_path):
    img = load_image(image_path)
    
    # 驗證集不做隨機裁切，直接 Resize 到目標大小
    img = tf.image.resize(img, size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    
    # 確保數值在 [0, 1] (通常 load_image 已經做了，但保險起見)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    return seq_emb, img

def dataset_generator(filenames, batch_size, train_size, save_embeding_path, root_dir):
    # load the training data
    df = pd.read_pickle(filenames)
    seq_emb = np.load(save_embeding_path)
    seq_emb = np.squeeze(seq_emb, axis=1) 
    
    # 路徑處理
    df['ImagePath'] = df['ImagePath'].str.replace('^\\.', '', regex=True)
    image_paths = root_dir + df['ImagePath'].values
    
    # 切分資料索引
    N, _, _ = seq_emb.shape
    split_idx = int(N * train_size)
    
    seq_train = seq_emb[:split_idx]        
    img_train = image_paths[:split_idx]
    
    seq_val = seq_emb[split_idx:]          
    img_val = image_paths[split_idx:]
    
    # --- 建立 Training Dataset (使用 process_train_data) ---
    assert seq_train.shape[0] == img_train.shape[0]
    train_data = tf.data.Dataset.from_tensor_slices((seq_train, img_train))
    # 這裡使用 process_train_data
    train_data = train_data.map(process_train_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(len(seq_train)).batch(batch_size, drop_remainder=True)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # --- 建立 Validation Dataset (使用 process_val_data) ---
    assert seq_val.shape[0] == img_val.shape[0]
    val_data = tf.data.Dataset.from_tensor_slices((seq_val, img_val))
    # 這裡使用 process_val_data (不做增強，確保驗證公平性)
    val_data = val_data.map(process_val_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_data = val_data.shuffle(len(seq_val)).batch(batch_size, drop_remainder=True)
    val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # --- 建立 Full Dataset (通常不需要增強，或視需求而定) ---
    # 這裡我預設全量數據不做增強，如果這是用來最終訓練的，可以改用 process_train_data
    dataset = tf.data.Dataset.from_tensor_slices((seq_emb, image_paths))
    dataset = dataset.map(process_val_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(seq_emb)).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset, train_data, val_data
