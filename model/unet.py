import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .common import PaddedConv2D, GEGLU, apply_seq, td_dot

class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(out_channels),
        ]
        self.out_layers = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_head):
        super().__init__()
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head**-0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [keras.layers.Dense(n_heads * d_head)]

    def call(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = keras.layers.Permute((2, 1, 3))(q)  # (bs, num_heads, time, head_size)
        k = keras.layers.Permute((2, 3, 1))(k)  # (bs, num_heads, head_size, time)
        v = keras.layers.Permute((2, 1, 3))(v)  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attention = td_dot(weights, v)
        attention = keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, num_heads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        return apply_seq(h_, self.to_out)


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        x, context = inputs
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels, n_heads, d_head):
        super().__init__()
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        assert channels == n_heads * d_head
        self.proj_in = PaddedConv2D(n_heads * d_head, 1)
        self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head)]
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, inputs):
        x, context = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + x_in


class Downsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.op = PaddedConv2D(channels, 3, stride=2, padding=1)

    def call(self, x):
        return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class UNetModelSmall(keras.models.Model):
    """
    縮小版 UNet
    """
    def __init__(self):
        super().__init__()

        # 時間 embedding MLP
        self.time_mlp = [
            layers.Dense(512),
            layers.Activation("swish"),
            layers.Dense(512),
        ]

        # ================= Encoder =================
        # Input: [16, 16, 4]
        self.input_blocks = [
            # --- Level 0: 16x16 (Channel: 128) ---
            # Enc 0: Input Conv
            [PaddedConv2D(128, kernel_size=3, padding=1)],

            # Enc 1: [16, 16, 128]
            [ResBlock(128, 128), SpatialTransformer(128, 8, 16)], # Heads=4, Dim=16

            # Enc 2: [16, 16, 128]
            [ResBlock(128, 128), SpatialTransformer(128, 8, 16)],

            # Enc 3: Downsample 16x16 -> 8x8
            [Downsample(128)],

            # --- Level 1: 8x8 (Channel: 256) ---
            # Enc 4: [8, 8, 128] -> [8, 8, 256]
            [ResBlock(128, 256), SpatialTransformer(256, 8, 32)],

            # Enc 5: [8, 8, 256]
            [ResBlock(256, 256), SpatialTransformer(256, 8, 32)],

            # Enc 6: Downsample 8x8 -> 4x4
            [Downsample(256)],

            # --- Level 2: 4x4 (Channel: 512) ---
            # Enc 7: [4, 4, 256] -> [4, 4, 512]
            [ResBlock(256, 512), SpatialTransformer(512, 8, 64)],

            # Enc 8: [4, 4, 512]
            [ResBlock(512, 512), SpatialTransformer(512, 8, 64)],

            # Enc 9: Downsample 4x4 -> 2x2
            [Downsample(512)],

            # --- Level 3: 2x2 (Channel: 512) Bottleneck ---
            # Enc 10: [2, 2, 512]
            [ResBlock(512, 512)],

            # Enc 11: [2, 2, 512]
            [ResBlock(512, 512)],
        ]

        # ================= Middle (2x2, 512 ch) =================
        self.middle_block = [
            ResBlock(512, 512),
            SpatialTransformer(512, 8, 64),
            ResBlock(512, 512),
        ]

        # ================= Decoder =================
        # saved_inputs stack 順序對應 Encoder 0~11
        
        self.output_blocks = [
            # --- Level 3: 2x2 (Skip Enc 11, 10, 9) ---
            # Dec 0 (Skip Enc 11): (512 + 512) -> 512
            [ResBlock(1024, 512)],

            # Dec 1 (Skip Enc 10): (512 + 512) -> 512
            [ResBlock(1024, 512)],

            # Dec 2 (Skip Enc 9): (512 + 512) -> 512 -> Up to 4x4
            [ResBlock(1024, 512), Upsample(512)],

            # --- Level 2: 4x4 (Skip Enc 8, 7, 6) ---
            # Dec 3 (Skip Enc 8): (512 + 512) -> 512
            [ResBlock(1024, 512), SpatialTransformer(512, 8, 64)],

            # Dec 4 (Skip Enc 7): (512 + 512) -> 512 -> Reduce to 256
            [ResBlock(1024, 512), SpatialTransformer(512, 8, 64)],

            # Dec 5 (Skip Enc 6): (256 + 256) -> 256 -> Up to 8x8
            [ResBlock(768, 512),
             SpatialTransformer(512, 8, 64),
             Upsample(512)],

            # --- Level 1: 8x8 (Skip Enc 5, 4, 3) ---
            # Dec 6 (Skip Enc 5): (256 + 256) -> 256
            [ResBlock(768, 256), SpatialTransformer(256, 8, 32)],

            # Dec 7 (Skip Enc 4): (256 + 256) -> 256 -> Reduce to 128
            [ResBlock(512, 256), SpatialTransformer(256, 8, 32)],

            # Dec 8 (Skip Enc 3): (128 + 128) -> 128 -> Up to 16x16
            [ResBlock(384, 256),
             SpatialTransformer(256, 8, 32),
             Upsample(256)],

            # --- Level 0: 16x16 (Skip Enc 2, 1, 0) ---
            # Dec 9 (Skip Enc 2): (128 + 128) -> 128
            [ResBlock(384, 128), SpatialTransformer(128, 8, 16)],

            # Dec 10 (Skip Enc 1): (128 + 128) -> 128
            [ResBlock(256, 128), SpatialTransformer(128, 8, 16)],

            # Dec 11 (Skip Enc 0): (128 + 128) -> 128
            [ResBlock(256, 128), SpatialTransformer(128, 8, 16)],
        ]

        # ================= Output =================
        self.out = [
            layers.GroupNormalization(epsilon=1e-5),
            layers.Activation("swish"),
            PaddedConv2D(4, kernel_size=3, padding=1),  # Output: [B, 16, 16, 4]
        ]

    def call(self, inputs):
        """
        inputs:
          x:       [B, 16, 16, 4]
          t_emb:   [B, 512]
          context: [B, 77, 768]
        """
        x, t_emb, context = inputs

        # time MLP
        emb = apply_seq(t_emb, self.time_mlp)

        def apply_layer(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []

        # ---------- Encoder ----------
        for block in self.input_blocks:
            for layer in block:
                x = apply_layer(x, layer)
            saved_inputs.append(x)

        # ---------- Middle ----------
        for layer in self.middle_block:
            x = apply_layer(x, layer)

        # ---------- Decoder ----------
        for block in self.output_blocks:
            skip = saved_inputs.pop()
            # Skip connection concatenation
            x = tf.concat([x, skip], axis=-1)
            for layer in block:
                x = apply_layer(x, layer)

        # ---------- Output ----------
        x = apply_seq(x, self.out)
        return x
