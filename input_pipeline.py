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
    
    # 自動處理維度
    # 情況 1: (N, 1, 77, 768) -> squeeze -> (N, 77, 768)
    if seq_emb.ndim == 4 and seq_emb.shape[1] == 1:
        seq_emb = np.squeeze(seq_emb, axis=1)
    # 情況 2: (N, 5, 1, 77, 768) -> squeeze -> (N, 5, 77, 768) (您的檔案是這個形狀)
    elif seq_emb.ndim == 5 and seq_emb.shape[2] == 1:
        seq_emb = np.squeeze(seq_emb, axis=2)
    
    # 檢查是否為多 Caption 模式: (N, Caps, 77, 768)
    has_multiple_captions = (seq_emb.ndim == 4 and seq_emb.shape[1] > 1)

    # 路徑處理
    df['ImagePath'] = df['ImagePath'].str.replace('^\\.', '', regex=True)
    image_paths = root_dir + df['ImagePath'].values
    
    # 切分資料索引
    N = seq_emb.shape[0]
    split_idx = int(N * train_size)
    
    seq_train = seq_emb[:split_idx]        
    img_train = image_paths[:split_idx]
    
    seq_val = seq_emb[split_idx:]          
    img_val = image_paths[split_idx:]
    
    # --- 定義 Mapping Wrapper ---
    if has_multiple_captions:
        # 如果有多個 caption，訓練時隨機選一個
        def train_map_wrapper(seq_emb_all, image_path):
            num_caps = tf.shape(seq_emb_all)[0]
            idx = tf.random.uniform([], 0, num_caps, dtype=tf.int32)
            seq_emb = seq_emb_all[idx]
            return process_train_data(seq_emb, image_path)
            
        def val_map_wrapper(seq_emb_all, image_path):
            # 驗證時固定選第一個，保持穩定
            seq_emb = seq_emb_all[0]
            return process_val_data(seq_emb, image_path)
    else:
        # 單一 caption 直接傳遞
        train_map_wrapper = process_train_data
        val_map_wrapper = process_val_data

    # --- 建立 Training Dataset ---
    assert seq_train.shape[0] == img_train.shape[0]
    train_data = tf.data.Dataset.from_tensor_slices((seq_train, img_train))
    train_data = train_data.map(train_map_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(len(seq_train)).batch(batch_size, drop_remainder=True)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # --- 建立 Validation Dataset ---
    assert seq_val.shape[0] == img_val.shape[0]
    val_data = tf.data.Dataset.from_tensor_slices((seq_val, img_val))
    val_data = val_data.map(val_map_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_data = val_data.shuffle(len(seq_val)).batch(batch_size, drop_remainder=True)
    val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # --- 建立 Full Dataset ---
    dataset = tf.data.Dataset.from_tensor_slices((seq_emb, image_paths))
    dataset = dataset.map(val_map_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE) # 使用 val 邏輯
    dataset = dataset.shuffle(len(seq_emb)).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset, train_data, val_data
