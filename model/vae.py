import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .common import PaddedConv2D, apply_seq

class ResnetBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(out_channels, 3, padding=1)
        self.norm2 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(out_channels, 3, padding=1)
        self.nin_shortcut = (
            PaddedConv2D(out_channels, 1)
            if in_channels != out_channels
            else lambda x: x
        )

    def call(self, x):
        h = self.norm1(x)
        h = keras.activations.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = keras.activations.swish(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(channels, 1)
        self.k = PaddedConv2D(channels, 1)
        self.v = PaddedConv2D(channels, 1)
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, x):
        # x: [B, H, W, C]
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b = tf.shape(q)[0]
        h_ = tf.shape(q)[1]
        w_ = tf.shape(q)[2]
        c = tf.shape(q)[3]
        hw = h_ * w_

        def reshape_hw(t):
            return tf.reshape(t, [b, hw, c])

        q_flat = reshape_hw(q)
        k_flat = reshape_hw(k)
        v_flat = reshape_hw(v)

        attn_logits = tf.matmul(q_flat, k_flat, transpose_b=True)
        attn_logits = attn_logits * tf.cast(c, tf.float32) ** -0.5
        attn = tf.nn.softmax(attn_logits, axis=-1)

        out_flat = tf.matmul(attn, v_flat)
        out = tf.reshape(out_flat, [b, h_, w_, c])
        out = self.proj_out(out)
        return x + out


class Encoder(keras.Model):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.down = keras.Sequential(
            [
                # 64x64 -> 64x64
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128, 128),
                ResnetBlock(128, 128),

                # 64 -> 32
                PaddedConv2D(128, 3, padding=(0,1), stride=2),

                ResnetBlock(128, 256),
                ResnetBlock(256, 256),

                # 32 -> 16
                PaddedConv2D(256, 3, padding=(0,1), stride=2),

                ResnetBlock(256, 512),
                ResnetBlock(512, 512),

                # 16 -> 8
                PaddedConv2D(512, 3, padding=(0,1), stride=2),

                ResnetBlock(512, 512),
                ResnetBlock(512, 512),

                # mid block with attention at 8x8
                ResnetBlock(512, 512),
                AttentionBlock(512),
                ResnetBlock(512, 512),

                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                
            ]
        )
        # 輸出 2 * latent_channels，前半是 mu，後半是 logvar
        self.conv_out1 = PaddedConv2D(latent_channels * 2, 3, padding=1)
        self.conv_out2 = PaddedConv2D(latent_channels * 2, 1)

        self.latent_channels = latent_channels

    def call(self, x, training=False):
        h = self.down(x, training=training)
        h = self.conv_out1(h)
        h = self.conv_out2(h)  # [B, 8, 8, 2C]
        mu, logvar = tf.split(h, num_or_size_splits=2, axis=-1)
        return mu, logvar


class Decoder(keras.Model):
    def __init__(self, latent_channels=4):
        super().__init__()

        self.latent_channels = latent_channels

        self.in_conv = keras.Sequential([
            PaddedConv2D(4, 1),
            PaddedConv2D(512, 3, padding=1),
        ])

        self.mid = keras.Sequential(
            [
                ResnetBlock(512, 512),
                AttentionBlock(512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
            ]
        )

        self.up = keras.Sequential(
            [
                # 8 -> 16
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),

                # 16 -> 32
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 256),
                ResnetBlock(256, 256),
                ResnetBlock(256, 256),

                # 32 -> 64
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(256, 128),
                ResnetBlock(128, 128),
                ResnetBlock(128, 128),

                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ]
        )
        

    def call(self, z, training=False):
        # z: [B, 8, 8, latent_channels]
        # 先升到 512 channels
        h = self.in_conv(z)
        h = self.mid(h, training=training)
        x_recon = self.up(h, training=training)  # [B, 64, 64, 3]
        return x_recon


class AutoencoderKL(keras.Model):
    def __init__(self, encoder, decoder, latent_channels=4, scaling_factor=0.18215):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        
    def encode(self, x, training=False):
        mu, logvar = self.encoder(x, training=training)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=False):
        if training:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std))
            z = mu + eps * std
            z = z
        else:
            z = mu
        return z
    
    def decode(self, z, training=False):
        x_recon = self.decoder(z, training=training)
        return x_recon
    
    def call(self, x, training=False):
        mu, logvar = self.encode(x, training=training)
        z = self.reparameterize(mu, logvar, training=training)
        x_recon = self.decode(z, training=training)
        return x_recon, mu, logvar, z
    
    @staticmethod
    def kl_loss(mu, logvar):
        """KL 散度損失"""
        kl = 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(mu) + tf.exp(logvar) - logvar - 1.0,
                axis=[1, 2, 3],
            )
        )
        return kl
    
    @staticmethod
    def reconstruction_loss(x, x_recon, l1_weight=1.0, l2_weight=0.0):
        """
        增強型重建損失
        """
        l1 = tf.reduce_mean(tf.abs(x - x_recon))
        l2 = tf.reduce_mean(tf.square(x - x_recon))
        
        return (
            l1_weight * l1 + 
            l2_weight * l2 
        )


class SDPatchGANDiscriminator(keras.Model):
    """
    Stable Diffusion / VQGAN 使用的 NLayerDiscriminator 的 Keras 版實作。
    """

    def __init__(self,
                 input_channels=3,
                 base_filters=64,   # ndf
                 n_layers=3,        # 卷積層數（不含最後輸出層）
                 use_actnorm=False  # SD 一般是 False => BatchNorm
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.base_filters = base_filters
        self.n_layers = n_layers
        self.use_actnorm = use_actnorm

        self.model = self._build_model()

    def _norm_layer(self):
        # SD 預設是 BatchNorm2d
        return layers.BatchNormalization

    def _build_model(self):
        kw = 4
        pad = "same"
        Norm = self._norm_layer()

        seq = []

        # --- 第一層：Conv + LeakyReLU（沒有 normalization） ---
        nf_mult = 1
        seq.append(layers.Conv2D(
            self.base_filters * nf_mult,
            kernel_size=kw,
            strides=2,
            padding=pad,
            use_bias=True      # 原始 PyTorch 版第一層有 bias
        ))
        seq.append(layers.LeakyReLU(0.2))

        # --- 中間層：逐層加深，stride=2，下採樣 ---
        nf_mult_prev = nf_mult
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # 最多乘到 8

            seq.append(layers.Conv2D(
                self.base_filters * nf_mult,
                kernel_size=kw,
                strides=2,
                padding=pad,
                use_bias=False   # 後面接 norm，原始實作會關掉 bias
            ))
            seq.append(Norm())
            seq.append(layers.LeakyReLU(0.2))

        # --- 倒數第二層：stride=1，維持空間尺寸 ---
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.n_layers, 8)

        seq.append(layers.Conv2D(
            self.base_filters * nf_mult,
            kernel_size=kw,
            strides=1,
            padding=pad,
            use_bias=False
        ))
        seq.append(Norm())
        seq.append(layers.LeakyReLU(0.2))

        # --- 最後輸出 1 channel 的 patch score map ---
        seq.append(layers.Conv2D(
            1,
            kernel_size=kw,
            strides=1,
            padding=pad,
            use_bias=True
        ))

        return keras.Sequential(seq, name="sd_patchgan_discriminator")

    def call(self, x, training=False):
        return self.model(x, training=training)

class LPIPSLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用 VGG19 的特定層來計算感知損失 (Perceptual Loss)
        vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
        # 鎖定參數，不訓練 VGG
        vgg.trainable = False
        
        # 選擇多個特徵層
        outputs = [
            vgg.get_layer('block1_conv2').output,
            vgg.get_layer('block2_conv2').output,
            vgg.get_layer('block3_conv4').output,
            vgg.get_layer('block4_conv4').output,
            vgg.get_layer('block5_conv4').output
        ]
        self.vgg_model = keras.Model(vgg.input, outputs)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def preprocess(self, x):
        # 假設輸入 x 在 [-1, 1] 之間
        # VGG 需要輸入在 [0, 1] 區間，然後做 ImageNet Normalize
        x = (x + 1.0) * 0.5  # [-1, 1] -> [0, 1]
        
        # (Batch, H, W, 3) - (3,) -> Broadcasting
        x = (x - self.mean) / self.std
        return x

    def call(self, real, fake):
        # 確保形狀正確
        real_features = self.vgg_model(self.preprocess(real))
        fake_features = self.vgg_model(self.preprocess(fake))
        
        loss = 0.0
        # 加總所有層的 L1 距離 (或 L2)
        for r, f in zip(real_features, fake_features):
            loss += tf.reduce_mean(tf.abs(r - f))
            
        return loss
