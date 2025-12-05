# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import CosineDecay
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom Modules
from model.vae import Encoder, Decoder, AutoencoderKL, SDPatchGANDiscriminator
from model.unet import UNetModelSmall
from model.text_encoder import TextEncoderWrapper
import input_pipeline
from trainer import VAETrainer, LatentDiffusionTrainer

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %%
ROOT = '/kaggle/input/competition-3/2025-datalab-cup3-reverse-image-caption'
VOCAB = '/dictionary/id2Word.npy'
WORD2ID = '/dictionary/word2Id.npy'
ID2WORD = '/dictionary/id2Word.npy'
TRAIN = '/dataset/text2ImgData.pkl'
TEST = '/dataset/testData.pkl'

# %%
train_path = ROOT+TRAIN
vocab_path = ROOT+VOCAB
word2_id_path = ROOT+WORD2ID
id2_word_path = ROOT+ID2WORD

# %%
vocab = np.load(vocab_path)
print('there are {} vocabularies in total'.format(len(vocab)))
word2Id_dict = dict(np.load(word2_id_path))
id2word_dict = dict(np.load(id2_word_path))
print('Word to id mapping, for example: %s -> %s' % ('flower', word2Id_dict['flower']))
print('Id to word mapping, for example: %s -> %s' % ('1', id2word_dict['1']))
print('Tokens: <PAD>: %s; <RARE>: %s' % (word2Id_dict['<PAD>'], word2Id_dict['<RARE>']))

# %%
df_train = pd.read_pickle(train_path)
print(df_train.head(5))
print(df_train.shape)

# %%
text = "the flower shown has yellow anther red pistil and bright red petals."
# Note: sent2IdList needs word2Id_dict passed to it
print(text)
print(input_pipeline.sent2IdList(text, word2Id_dict))

# %%
sample = ['9', '1', '82', '5', '11', '70', '20', '31', '3', '29', '20', '2', '5427', '5427', '5427', '5427', '5427', '5427', '5427', '5427']
# Note: id2Sent needs id2word_dict
sample_text = input_pipeline.id2Sent(sample, id2word_dict)
print(sample_text)

# %% [markdown]
# ## TEXT ENCODER

# %%
MAX_LEN = 77
text_encoder = TextEncoderWrapper()

# %%
sample = ['9', '1', '82', '5', '11', '70', '20', '31', 
          '3', '29', '20', '2', '5427', '5427', '5427', '5427', 
          '5427', '5427', '5427', '5427']
sample_emb = text_encoder.id2clip(sample, id2word_dict)
print(sample_emb.shape)

# %%
save_embeding_path = "/kaggle/working/seq_emb_multi.npy"
max_caption_len = 5

def preprocess_captions_all(filenames, save_path, max_len):
    df = pd.read_pickle(filenames)
    captions_list = df["Captions"].values     
    seq_emb_list = []
    for cap_list in tqdm(captions_list):
        emb_list = []
        for i,ids in enumerate(cap_list):
            if i>=max_len:
                break
            emb = text_encoder.id2clip(ids, id2word_dict)                
            emb_list.append(emb)
        emb_arr = np.stack(emb_list, axis=0)   
        seq_emb_list.append(emb_arr)
    seq_emb_arr = np.stack(seq_emb_list, axis=0)
    print("saved:", seq_emb_arr.shape)        
    np.save(save_path, seq_emb_arr)

# Comment out if not needed or already run
# preprocess_captions_all(train_path, save_embeding_path,max_caption_len)

# %%
BATCH_SIZE = 16
# Note: dataset_generator now needs save_embeding_path and root_dir passed
dataset, train_data, valid_data = input_pipeline.dataset_generator(
    train_path, BATCH_SIZE, 0.8, save_embeding_path, ROOT
)

# %% [markdown]
# ## VAE

# %%
loss_start = 50001 # Variable used for VAE trainer config

decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
            file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
        )

encoder_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
    file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
)

decoder = Decoder()
latent = keras.layers.Input((16,16,4))
decoder = keras.models.Model(latent, decoder(latent))
decoder.load_weights(decoder_weights_fpath)

encoder = Encoder()
inp_img = keras.layers.Input((128,128,3))
encoder = keras.models.Model(inp_img,encoder(inp_img))
encoder.load_weights(encoder_weights_fpath)

# %%
h_disc_start= 0
h_kl_weight=1e-6 
h_perc_weight=0.1
h_disc_weight=0.05
vae = AutoencoderKL(encoder,decoder,latent_channels=4, scaling_factor=0.18215)
discriminator = SDPatchGANDiscriminator(
    input_channels=3,
    base_filters=64,
    n_layers=3  # SD, VQGAN 常用設定
)
trainer = VAETrainer(vae, discriminator,disc_start=h_disc_start,kl_weight=h_kl_weight,perc_weight=h_perc_weight,disc_weight=h_disc_weight)

base_lr = 4.5e-6
lr = base_lr * BATCH_SIZE
vae_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
disc_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

trainer.compile(
    vae_optimizer=vae_opt,
    disc_optimizer=disc_opt
)

# %%
weight_path = {}
weight_path['encoder'] = '/kaggle/working/encoder.weights.h5'
weight_path['decoder'] = '/kaggle/working/decoder.weights.h5'

# %%
def show_vae_reconstructions(autoencoder, dataset, num_images=8):
    for _ ,img_batch in dataset.take(1):
        imgs = img_batch[:num_images]          # [N, 64, 64, 3], in [0,1]
        x_in = imgs * 2.0 - 1.0
        x_recon, mu, logvar, z = autoencoder(x_in, training=False)
        x_recon = (x_recon + 1.0) / 2.0
        imgs_vis = tf.clip_by_value(imgs, 0.0, 1.0)
        recon_vis = tf.clip_by_value(x_recon, 0.0, 1.0)
        plt.figure(figsize=(2 * num_images, 4))
        for i in range(num_images):
            # 上排：原圖
            plt.subplot(2, num_images, i + 1)
            plt.imshow(imgs_vis[i].numpy())
            plt.axis("off")
            if i == 0:
                plt.ylabel("Original", fontsize=12)

            # 下排：重建圖
            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(recon_vis[i].numpy())
            plt.axis("off")
            if i == 0:
                plt.ylabel("Reconstructed", fontsize=12)

        plt.tight_layout()
        plt.show()
        break  # 只看第一個 batch 就好

# %%
# Train VAE (Commented out for safety as requested)
# Epochs = 50
# start_epoch = 0
# for epoch in range(start_epoch,Epochs):
#     # ... training loop code ...
#     pass

# %%
vae.encoder.trainable = False
vae.decoder.trainable = False
vae.trainable = False

# %%
def calculate_latent_scale(vae_encoder, dataset):
    all_latents =[]
    for _, img_batch in tqdm(dataset):
        
        mean, log_var = vae_encoder(img_batch)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + epsilon * tf.exp(0.5 * log_var)
        all_latents.append(z.numpy())

    all_latents = np.concatenate(all_latents, axis=0) # Shape: [Total_N, 8, 8, 4]
    
    # 5. 計算整體的標準差 (Standard Deviation)
    mean = np.mean(all_latents)
    std = np.std(all_latents)
    return mean,std

# mean, scale_factor = calculate_latent_scale(vae.encoder,dataset)
# print(mean, scale_factor)

# %% [markdown]
# ## UNET

# %%
TOTAL_EPOCHS =  50
EPOCH_STEPS = 460
TOTAL_STEPS = TOTAL_EPOCHS * EPOCH_STEPS
WARMUP_RATIO = 0.1
WARMUP_STEPS = int(TOTAL_STEPS*WARMUP_RATIO )
INITIAL_LR = 1e-4

unet = UNetModelSmall()
ema_unet = UNetModelSmall()

# ==========================================
# [修改] 使用 keras.Input 強制 Build
# ==========================================
# 定義輸入形狀 (Batch Size 不用寫，Keras 會自動處理)
x_in = keras.Input(shape=(16, 16, 4))       # 對應 Latents
t_in = keras.Input(shape=(512,))            # 對應 Timestep Embedding
c_in = keras.Input(shape=(77, 768))         # 對應 Context (Text) Embedding

# 呼叫模型以建立權重 (觸發 build)
unet([x_in, t_in, c_in])
ema_unet([x_in, t_in, c_in])

# ==========================================
# 現在可以安全複製權重了
# ==========================================
print("Weights initialized. Copying to EMA model...")
ema_unet.set_weights(unet.get_weights())
diffusion_model = LatentDiffusionTrainer(unet,ema_unet,vae.encoder,vae.decoder)

lr_schedule = CosineDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=TOTAL_STEPS, 
    warmup_steps=WARMUP_STEPS
)
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule, 
    weight_decay=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
loss_fn = keras.losses.MeanSquaredError()
diffusion_model.compile(optimizer=optimizer, loss_fn=loss_fn)

# %%
checkpoint_path = f"/kaggle/working/checkpoints/tf_checkpoint.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# %%
# diffusion_model.load_weights(checkpoint_path)

# %%
# diffusion_model.fit(
#     dataset,
#     validation_data=valid_data,
#     epochs=50,
#     # Note: LambdaCallback might need adjustment if plot_images requires arguments 
#     # that are not provided by on_epoch_end automatically. 
#     # However, in the trainer, plot_images arguments have defaults or hardcodes except valid_data.
#     # We need to wrap it.
#     callbacks=[
#         keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: diffusion_model.plot_images(valid_data)),
#         checkpoint_callback,
#     ],
# )
