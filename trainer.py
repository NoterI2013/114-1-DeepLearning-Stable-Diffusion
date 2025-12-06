import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from model.vae import LPIPSLayer

class VAETrainer(keras.Model):
    def __init__(self, vae, discriminator, 
                 disc_start=50001,  # <--- 關鍵：前 50000 步不訓練 GAN
                 kl_weight=1e-6, 
                 perc_weight=0.1,
                 disc_weight=0.5):  # CompVis 原始設定通常是 0.5
        super().__init__()
        self.vae = vae
        self.discriminator = discriminator
        self.lpips = LPIPSLayer()
        
        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.perc_weight = perc_weight
        self.disc_weight = disc_weight
        
        # 追蹤步數
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.perc_loss_tracker = keras.metrics.Mean(name="perc_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    def compile(self, vae_optimizer, disc_optimizer):
        super().compile()
        self.vae_optimizer = vae_optimizer
        self.disc_optimizer = disc_optimizer

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.perc_loss_tracker,
            self.gen_loss_tracker,
            self.disc_loss_tracker,
        ]
    @tf.function
    def train_step(self, data):
        # 處理數據輸入，如果是 tuple (x, y) 取 x
        if isinstance(data, tuple):
            data = data[0]
        
        real_imgs = data # 假設已經是 [-1, 1]

        with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
            # 1. VAE Forward
            # 這裡假設 AutoencoderKL 回傳 (x_recon, mu, logvar, z)
            recon_imgs, mu, logvar, _ = self.vae(real_imgs, training=True)

            # --- Reconstruction Loss (L1) ---
            recon_loss = tf.reduce_mean(tf.abs(real_imgs - recon_imgs))
            
            # --- Perceptual Loss (LPIPS) ---
            p_loss = self.lpips(real_imgs, recon_imgs)
            
            # --- KL Loss ---
            # 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=[1, 2, 3])
            kl_loss = tf.reduce_mean(kl_loss)

            # --- Generator Loss (GAN) ---
            # 判斷是否達到開啟 GAN 的步數
            start_gan = self.global_step >= self.disc_start
            
            if start_gan:
                # 這裡計算 generator 想要騙過 discriminator 的 loss
                logits_fake = self.discriminator(recon_imgs, training=True)
                g_loss = -tf.reduce_mean(logits_fake)
                
                # 根據 CompVis 實作，這裡通常會有一個 adaptive weight (lambda)，
                # 為了簡化，我們先用固定的 disc_weight，若要進階需計算最後一層梯度比值。
                disc_factor = self.disc_weight 
            else:
                g_loss = 0.0
                disc_factor = 0.0

            # VAE Total Loss
            vae_loss = (
                recon_loss + 
                (self.perc_weight * p_loss) + 
                (self.kl_weight * kl_loss) + 
                (disc_factor * g_loss)
            )

            # --- Discriminator Loss ---
            if start_gan:
                logits_real = self.discriminator(real_imgs, training=True)
                # 這裡一定要用 detached (stop_gradient) 的 recon_imgs，否則梯度會流回 VAE
                logits_fake_detached = self.discriminator(tf.stop_gradient(recon_imgs), training=True)
                
                # Hinge Loss
                d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - logits_real))
                d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + logits_fake_detached))
                disc_loss = 0.5 * (d_loss_real + d_loss_fake)
            else:
                disc_loss = 0.0

        # 2. Update VAE
        vae_grads = tape.gradient(vae_loss, self.vae.trainable_weights)
        # 加上 Gradient Clipping 防止梯度爆炸
        vae_grads = [tf.clip_by_norm(g, 1.0) for g in vae_grads] 
        self.vae_optimizer.apply_gradients(zip(vae_grads, self.vae.trainable_weights))

        # 3. Update Discriminator (Only if started)
        if start_gan:
            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]
            self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

        # 更新步數
        self.global_step.assign_add(1)

        # Update metrics
        self.loss_tracker.update_state(vae_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perc_loss_tracker.update_state(p_loss)
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {
            "loss": self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perc_loss": self.perc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            "step": self.global_step
        }

image_size = 128
kid_image_size = 156

class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        # Use TensorFlow functions instead of ops
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype="float32")
        return (tf.matmul(features_1, tf.transpose(features_2)) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype="float32")
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


class LatentDiffusionTrainer(keras.Model):
    def __init__(self, unet, ema_unet, vae_encoder, vae_decoder, latent_scale=0.1825):
        super().__init__()
        self.unet = unet
        self.ema_unet = ema_unet
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.timesteps = 1000
        self.latent_scale = latent_scale
        self.built = True
        self.ema_decay= 0.99
        self.vae_encoder.trainable = False
        self.vae_decoder.trainable = False
        
        min_signal_rate = 0.02
        max_signal_rate = 0.95
        eta_max = np.arccos(max_signal_rate)
        eta_min = np.arccos(min_signal_rate)
        self.eta_start = tf.constant(eta_max,dtype='float32')
        self.eta_end = tf.constant(eta_min,dtype='float32')

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def timestep_embedding(self, timesteps, dim=512, max_period=10000.0):
        half = dim // 2
        freqs = tf.math.exp(
            -tf.math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.cast(timesteps, tf.float32)[:, None] * freqs[None, :]
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], axis=-1)
        return embedding
    
    def diffusion_schedule(self, diffusion_times):
        tf.debugging.assert_less_equal(diffusion_times, 1.0, 
                                   message="Diffusion times must be less than or equal to 1.0")
        tf.debugging.assert_greater_equal(diffusion_times, 0.0, 
                                      message="Diffusion times must be greater than or equal to 0.0")
        eta = (1.0-diffusion_times)*self.eta_start+(diffusion_times)*self.eta_end
        noise_rates, signal_rates = tf.sin(eta), tf.cos(eta)
        return noise_rates, signal_rates
    
    def denoise(self, noisy_images, t_emb, seq_emb, noise_rates, signal_rates, training):
        if training:
            network = self.unet
        else:
            network = self.ema_unet
        pred_noises = network([noisy_images,t_emb,seq_emb],training=training)
        pred_images = (noisy_images - noise_rates*pred_noises)/signal_rates
        return pred_noises, pred_images  
    
    
    def reverse_diffusion(self, initial_noise, diffusion_steps, seq_emb):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise
        
        for step in range(diffusion_steps):
            diffusion_times = max(1.0 - step * step_size, 0.0)
            next_diffusion_times = max(diffusion_times - step_size, 0.0)
    
            diffusion_times = tf.fill([num_images, 1, 1, 1], diffusion_times)
            next_diffusion_times = tf.fill([num_images, 1, 1, 1], next_diffusion_times)
          
            t_float = diffusion_times[0, 0, 0, 0]  
            t_index = tf.cast(
                tf.round(t_float * (self.timesteps - 1)), tf.int32
            )  
            timesteps = tf.fill([num_images], t_index)

            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            t_emb = self.timestep_embedding(timesteps)
    
            pred_noises, pred_images = self.denoise(
                next_noisy_images, t_emb, seq_emb,
                noise_rates, signal_rates,
                training=False,  
            )
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
            
        return pred_images
    
    def train_step(self, data):
        # 假設 data = (seq_emb, images)
        if isinstance(data, tuple) and len(data) == 2:
            seq_emb, images = data
        else:
            raise ValueError("Expected data to be a tuple (seq_emb, images)")

        batch_size = tf.shape(images)[0]
        images = images * 2.0 - 1.0
        
        mu, logvar = self.vae_encoder(images, training=False)
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        latents = mu + eps * std
        latents = latents * self.latent_scale

        timesteps = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.timesteps,
            dtype=tf.int32,
        )
        
        time_ratio = tf.cast(timesteps, tf.float32) / float(self.timesteps)
        time_ratio = tf.reshape(time_ratio, (-1, 1, 1, 1))
        
        noise_rates, signal_rates = self.diffusion_schedule(time_ratio)
        noises = tf.random.normal(shape=tf.shape(latents), dtype=latents.dtype)
        noisy_latents = signal_rates * latents + noise_rates * noises
        t_emb = self.timestep_embedding(timesteps)
        
        with tf.GradientTape() as tape:
            noise_pred, pred_latents = self.denoise(noisy_latents, t_emb, seq_emb,noise_rates,signal_rates,training=True)
            i_loss = self.loss_fn(latents, pred_latents) 
            n_loss = self.loss_fn(noises, noise_pred)

        gradients = tape.gradient(n_loss, self.unet.trainable_variables)
        clip_norm = 1.0
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        
        # EMA 更新
        for weight, ema_weight in zip(self.unet.weights, self.ema_unet.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1.0 - self.ema_decay) * weight)
            
        self.image_loss_tracker.update_state(i_loss)
        self.noise_loss_tracker.update_state(n_loss)
        return {m.name: m.result() for m in self.metrics[:-1]}

    def generate_images(self, batch_size, diffusion_steps, seq_emb):
        # 隨機初始 latent
        noisy_latents = tf.random.normal(
            shape=(batch_size, 16, 16, 4), dtype=tf.float32
        )
        pred_latents = self.reverse_diffusion(noisy_latents,diffusion_steps,seq_emb)
        pred_latents = pred_latents / self.latent_scale
        decoded_images = self.vae_decoder(pred_latents, training=False) # Changed to False because no global EMA var
        decoded_images = tf.clip_by_value((decoded_images / 2.0 + 0.5), 0.0, 1.0)
        return decoded_images

    def test_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            seq_emb, images = data
        else:
            raise ValueError("Expected data to be a tuple (seq_emb, images)")
        batch_size = tf.shape(images)[0]
        images = images * 2.0 - 1.0
        
        mu, logvar = self.vae_encoder(images, training=False)
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        latents = mu + eps * std
        latents = latents * self.latent_scale

        timesteps = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.timesteps,
            dtype=tf.int32,
        )
        time_ratio = tf.cast(timesteps, tf.float32) / float(self.timesteps)
        time_ratio = tf.reshape(time_ratio, (-1, 1, 1, 1))
        
        noise_rates, signal_rates = self.diffusion_schedule(time_ratio)
        noises = tf.random.normal(shape=tf.shape(latents), dtype=latents.dtype)
        noisy_latents = signal_rates * latents + noise_rates * noises
        t_emb = self.timestep_embedding(timesteps)

        noise_pred, pred_latents = self.denoise(noisy_latents, t_emb, seq_emb,noise_rates,signal_rates,training=False)
        i_loss = self.loss_fn(latents, pred_latents) 
        n_loss = self.loss_fn(noises, noise_pred)
        
        self.image_loss_tracker.update_state(i_loss)
        self.noise_loss_tracker.update_state(n_loss)
        images = (images + 1.0) / 2.0

        diffusion_steps = 10 # kid_diffusion_steps
        generated_images = self.generate_images(batch_size, diffusion_steps, seq_emb)

        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, valid_data, draw_diffusion_steps=30, num_rows=2, num_cols=8, figsize=(12, 5)):
        
        val_seq_emb_batch, _ = next(iter(valid_data))
        
        # Ensure batch compatibility by slicing the validation embedding
        target_batch_size = num_rows * num_cols
        if tf.shape(val_seq_emb_batch)[0] > target_batch_size:
            val_seq_emb_batch = val_seq_emb_batch[:target_batch_size]
        
        # Update batch_size in case validation batch is smaller than target
        batch_size = tf.shape(val_seq_emb_batch)[0]
        
        generated_samples = self.generate_images(batch_size, draw_diffusion_steps, val_seq_emb_batch)
        generated_samples = generated_samples.numpy()
        total_imgs = generated_samples.shape[0]
        if total_imgs < num_rows * num_cols and num_rows > 0:
            num_cols = total_imgs // num_rows

        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

        if num_rows == 1:
            ax = np.expand_dims(ax, axis=0)
        if num_cols == 1:
            ax = np.expand_dims(ax, axis=1)

        for i in range(min(total_imgs, num_rows * num_cols)):
            r = i // num_cols
            c = i % num_cols
            curr_ax = ax[r, c]
            curr_ax.imshow(generated_samples[i])
            curr_ax.axis("off")

        plt.tight_layout()
        plt.show()
