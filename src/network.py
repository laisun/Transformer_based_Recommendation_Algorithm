import tensorflow as tf

def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class Embedding(tf.keras.layers.Layer):
    def __init__(self, num_item, model_dim, seq_len, drop_rate, context_dim=10, pos_embed=True):
        super(Embedding, self).__init__()
        self.item_embedding = tf.keras.layers.Embedding(num_item+2, model_dim, mask_zero=False, input_length=seq_len)
        self.year_embedding = tf.keras.layers.Embedding(30, context_dim, mask_zero=False, input_length=seq_len) # [PAD], 1991 ~ 2019
        self.month_embedding =  tf.keras.layers.Embedding(13, context_dim, mask_zero=False, input_length=seq_len)
        self.day_embedding = tf.keras.layers.Embedding(32, context_dim, mask_zero=False, input_length=seq_len)
        self.hour_embedding =  tf.keras.layers.Embedding(25, context_dim, mask_zero=False, input_length=seq_len)
        self.context_dense = tf.keras.layers.Dense(model_dim, activation='linear')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(drop_rate)    

        self.pos_embedding = tf.keras.layers.Embedding(seq_len, model_dim, mask_zero=False, input_length=seq_len)
        self.pos = tf.range(0, seq_len)
        self.pos_embed = pos_embed

    def call(self, x, training):
        x_item, x_year, x_month, x_day, x_hour = x
        year_embedded = self.year_embedding(x_year)
        month_embedded = self.month_embedding(x_month)
        day_embedded = self.day_embedding(x_day)
        hour_embedded = self.hour_embedding(x_hour)
        context_embedded = tf.keras.layers.concatenate([year_embedded, month_embedded, day_embedded, hour_embedded], axis=-1)
        context_embedded = self.context_dense(context_embedded)
        
        item_embedded = self.item_embedding(x_item)
        embedded = item_embedded + context_embedded

        if self.pos_embed:
            pos_embedded = self.pos_embedding(self.pos)
            pos_embedded = pos_embedded[None,:,:]
            embedded += pos_embedded

        embedded = self.layernorm(embedded)
        embedded = self.dropout(embedded, training=training)
        return embedded


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.model_dim = model_dim
        self.depth = model_dim // num_head
        assert model_dim % num_head == 0

        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.dense = tf.keras.layers.Dense(model_dim)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)            
        return output, attention_weights


class PointWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, model_dim, dff):
        super(PointWiseFeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff)
        self.dense2 = tf.keras.layers.Dense(model_dim)
    
    def call(self, x):
        x = self.dense1(x)
        x = gelu(x)
        x = self.dense2(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_head, dff, drop_rate):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(model_dim, num_head)
        self.pwff = PointWiseFeedForward(model_dim, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        
    def call(self, x, mask, training):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        pwff_output = self.pwff(out1)  # (batch_size, input_seq_len, d_model)
        pwff_output = self.dropout2(pwff_output, training=training)
        out2 = self.layernorm2(out1 + pwff_output)  # (batch_size, input_seq_len, d_model)
        return out2

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, dff):
        super(OutputLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(model_dim)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.softmax = tf.keras.layers.Softmax()
        self.b = tf.Variable(tf.zeros([1]))
    
    def call(self, x, embedding_weights):
        x = gelu(self.dense(x))
        implicit = tf.matmul(x, embedding_weights, transpose_b=True) + self.b
        implicit = self.softmax(implicit)
        explicit = self.dense2(x)
        return implicit, explicit


class Network(tf.keras.Model):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.embedding = Embedding(config['num_item'], config['model_dim'], config['seq_len'], config['drop_rate'])
        self.decoder_layers = [DecoderLayer(config['model_dim'], config['num_head'], config['dff'], config['drop_rate']) for _ in range(config['num_layer'])]
        self.output_layer = OutputLayer(config['model_dim'], config['dff'])
    
    def call(self, x, training):
        x, (input_mask, output_mask) = x
        x = self.embedding(x)
        for i in range(self.config['num_layer']):
            x = self.decoder_layers[i](x, input_mask, training)
        implicit, explicit = self.output_layer(x, self.embedding.weights[0])
        return implicit, explicit