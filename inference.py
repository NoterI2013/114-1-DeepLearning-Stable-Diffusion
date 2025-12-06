# %%
import os

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Custom Modules
from model.vae import Encoder, Decoder, AutoencoderKL
from model.unet import UNetModelSmall
from trainer import LatentDiffusionTrainer

# %%
# --- Configuration ---
hparas = {
    'BATCH_SIZE': 32,
    'Z_DIM': 4, # Latent channels
    'IMAGE_SIZE': 128,
}

SEQ_EMB_PATH = './seqemb/seq_emb_test.npy'
TEST_DATA_PATH = './dataset/dataset/testData.pkl'
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/tf_checkpoint.weights.h5"
OUTPUT_DIR = './inference/demo'

# %%
# --- Hardware Setup ---
def setup_hardware():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs detected: {len(gpus)}")
        except RuntimeError as e:
            print(e)

# %%
setup_hardware()

# %%
# --- Model Loading ---
def get_vae_models():
    """Load and return VAE (Encoder/Decoder) with weights loaded."""
    # We use the same cached files as in cup3.py
    decoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
                file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
            )

    encoder_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
        file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
    )

    print("[VAE] Loading Decoder...")
    decoder = Decoder()
    latent = keras.layers.Input((16,16,4))
    decoder_model = keras.models.Model(latent, decoder(latent))
    decoder_model.load_weights(decoder_weights_fpath)

    print("[VAE] Loading Encoder...")
    encoder = Encoder()
    inp_img = keras.layers.Input((128,128,3))
    encoder_model = keras.models.Model(inp_img, encoder(inp_img))
    encoder_model.load_weights(encoder_weights_fpath)
    
    return encoder_model, decoder_model

# %%
def build_diffusion_model():
    print("[Model] Loading VAE...")
    encoder_model, decoder_model = get_vae_models()
    
    # Freeze VAE
    encoder_model.trainable = False
    decoder_model.trainable = False
    
    print("[Model] Initializing UNet...")
    unet = UNetModelSmall()
    ema_unet = UNetModelSmall()

    # Build UNet (Force build with dummy input)
    # Shape: x=(N, 16, 16, 4), t=(N,), c=(N, 77, 768)
    # Note: UNet call expects [x, t, c]
    x_in = keras.Input(shape=(16, 16, 4))
    t_in = keras.Input(shape=(512,)) # Timestep embedding
    c_in = keras.Input(shape=(77, 768))
    unet([x_in, t_in, c_in])
    ema_unet([x_in, t_in, c_in])
    
    print("[Model] Creating LatentDiffusionTrainer...")
    diffusion_model = LatentDiffusionTrainer(unet, ema_unet, encoder_model, decoder_model)
    
    # Compile dummy to allow loading weights if needed (though we use load_weights)
    # But usually creating the object is enough.
    
    return diffusion_model

# %%
def load_checkpoint(diffusion_model, path):
    if os.path.exists(path):
        print(f"[Checkpoint] Loading weights from {path}...")
        try:
            # Force KID build if necessary for loading (though for inference strictly it might not be needed if variables match)
             # But LatentDiffusionTrainer structure includes KID metric so load_weights might complain if layers don't match
            dummy_img = tf.zeros((1, 128, 128, 3))
            diffusion_model.kid.update_state(dummy_img, dummy_img)
            
            diffusion_model.load_weights(path, skip_mismatch=True)
            print("[Checkpoint] Loaded successfully.")
            return True
        except Exception as e:
            print(f"[Checkpoint] Failed to load: {e}")
            return False
    else:
        print(f"[Checkpoint] File not found: {path}")
        return False

# %%
# --- Data Loading ---
def testing_dataset_generator(batch_size):
    # Load Embeddings
    print(f"[Data] Loading embeddings from {SEQ_EMB_PATH}...")
    if not os.path.exists(SEQ_EMB_PATH):
        raise FileNotFoundError(f"{SEQ_EMB_PATH} not found.")
    
    captions_emb = np.load(SEQ_EMB_PATH)
    # Shape check: (819, 77, 768)
    print(f"[Data] Embeddings shape: {captions_emb.shape}")
    
    # Handle dimensions if needed (similar to input_pipeline)
    if captions_emb.ndim == 4 and captions_emb.shape[1] == 1:
        captions_emb = np.squeeze(captions_emb, axis=1)
    elif captions_emb.ndim == 5 and captions_emb.shape[2] == 1:
        captions_emb = np.squeeze(captions_emb, axis=2)

    captions_emb = captions_emb.astype(np.float32)

    # Load IDs
    print(f"[Data] Loading IDs from {TEST_DATA_PATH}...")
    data = pd.read_pickle(TEST_DATA_PATH)
    # Assuming 'ID' column exists or index is ID
    # User snippet: index = data['ID'].values
    if 'ID' in data.columns:
        index = data['ID'].values
    else:
        print("[Data] 'ID' column not found, using index as ID.")
        index = data.index.values
        
    index = np.asarray(index)
    
    dataset = tf.data.Dataset.from_tensor_slices((captions_emb, index))
    dataset = dataset.batch(batch_size)
    return dataset, len(index)

# %%
# --- Inference Functions ---
def inference_testset(diffusion_model, output_folder=OUTPUT_DIR):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("[Inference] Starting inference on full test set...")
    # Model is passed in, assuming weights are loaded
    
    dataset, num_samples = testing_dataset_generator(hparas['BATCH_SIZE'])
    
    start_time = time.time()
    
    for step, (batch_emb, batch_ids) in enumerate(dataset):
        print(f"Processing batch {step+1}...")
        batch_size = tf.shape(batch_emb)[0]
        
        # Generate images
        # diffusion_model.generate_images expects (batch_size, diffusion_steps, seq_emb)
        # We use a reasonable step count for inference (e.g., 50 or 100)
        diffusion_steps = 30 # Can be increased for better quality
        generated_images = diffusion_model.generate_images(batch_size, diffusion_steps, batch_emb)
        
        # Save images
        for i in range(batch_size):
            img = generated_images[i].numpy() # Already [0, 1] from generate_images
            img_id = batch_ids[i]
            
            # Format filename
            if isinstance(img_id, (int, np.integer)):
                 filename = f"inference_{img_id:04d}.jpg"
            else:
                 filename = f"inference_{img_id}.jpg"
                 
            save_path = os.path.join(output_folder, filename)
            plt.imsave(save_path, img)
            
    print(f"[Inference] Finished. Time taken: {time.time() - start_time:.2f}s")

# %%
def inference_one(idx_to_infer, diffusion_model=None, captions_emb=None, dataset_df=None):
    """
    Generate image for a single specific ID from the test set and display it.
    idx_to_infer can be an integer index or a specific ID string depending on dataset.
    """
    print(f"[Inference] Generating one image for ID/Index: {idx_to_infer}...")
    
    if diffusion_model is None:
        diffusion_model = build_diffusion_model()
        load_checkpoint(diffusion_model, CHECKPOINT_PATH)
    
    # Load Data Manually if not provided
    if captions_emb is None:
        print("[Inference] Loading embeddings (uncached)...")
        captions_emb = np.load(SEQ_EMB_PATH)
        # Fix dims
        if captions_emb.ndim == 4 and captions_emb.shape[1] == 1:
            captions_emb = np.squeeze(captions_emb, axis=1)
        elif captions_emb.ndim == 5 and captions_emb.shape[2] == 1:
            captions_emb = np.squeeze(captions_emb, axis=2)
            
    if dataset_df is None:
        print("[Inference] Loading dataframe (uncached)...")
        dataset_df = pd.read_pickle(TEST_DATA_PATH)
    
    data = dataset_df # alias
    
    # Find the row
    if 'ID' in data.columns:
        # Match ID
        # Convert to same type for comparison
        ids = data['ID'].values
        # Try finding index
        matches = np.where(ids == idx_to_infer)[0]
        if len(matches) > 0:
            target_idx = matches[0]
        else:
            print(f"ID {idx_to_infer} not found. Using as 0-based index.")
            target_idx = int(idx_to_infer)
    else:
        target_idx = int(idx_to_infer)
        
    if target_idx >= len(captions_emb):
        print(f"Index {target_idx} out of bounds.")
        return

    target_emb = captions_emb[target_idx]
    target_text = data.iloc[target_idx]['Captions'] if 'Captions' in data.columns else "Unknown Caption"
    
    # Add batch dimension
    target_emb = tf.expand_dims(target_emb, 0) # (1, 77, 768)
    
    # Generate
    diffusion_steps = 50
    generated_image = diffusion_model.generate_images(1, diffusion_steps, target_emb)
    generated_image = generated_image[0].numpy()
    
    # Display
    print(f"Caption: {target_text}")
    plt.figure(figsize=(6, 6))
    plt.imshow(generated_image)
    plt.title(f"Generated Image (ID: {idx_to_infer})\n{target_text[:50]}...")
    plt.axis('off')
    
    out_path = f"inference_{idx_to_infer}_single.jpg"
    plt.savefig(out_path)
    print(f"Saved inference result to {out_path}")
    plt.show()

# %%
if __name__ == "__main__":
    # 1. Unified Model Loading
    print("[Main] Initializing Model...")
    diffusion_model = build_diffusion_model()
    load_checkpoint(diffusion_model, CHECKPOINT_PATH)
    
    # 2. Choose Mode
    
    # --- Mode A: Full Test Set Inference ---
    # inference_testset(diffusion_model)
    
    # --- Mode B: Single/Loop Inference ---
    # Pre-load data for single inference efficiency (optional for full testset as it uses tf.dataset)
    print("[Main] Pre-loading data for single inference...")
    if os.path.exists(SEQ_EMB_PATH):
        all_captions_emb = np.load(SEQ_EMB_PATH)
        if all_captions_emb.ndim == 4 and all_captions_emb.shape[1] == 1:
            all_captions_emb = np.squeeze(all_captions_emb, axis=1)
        elif all_captions_emb.ndim == 5 and all_captions_emb.shape[2] == 1:
            all_captions_emb = np.squeeze(all_captions_emb, axis=2)
            
        all_dataset_df = pd.read_pickle(TEST_DATA_PATH)
        
        for i in range(10):
            inference_one(i, diffusion_model=diffusion_model, captions_emb=all_captions_emb, dataset_df=all_dataset_df)    
    else:
        print(f"Warning: {SEQ_EMB_PATH} not found. Cannot run inference loop.")    
