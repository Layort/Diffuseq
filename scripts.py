
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# decoder with attention mechanism
def attention_decoder(decoder_inputs,
                      initial_state,
                      encoder_states,
                      cell,
                      attn_mask,
                      initial_state_attention=False,
                      pointer=False):
    batch_size = encoder_states.size(0)
    attn_size = encoder_states.size(2)
    # batch_size = encoder_states.get_shape()[0].value
    # attn_size = encoder_states.get_shape()[2].value

    # encoder_states = tf.expand_dims(encoder_states, axis=2)
    encoder_states = encoder_states.unsqueeze(2)

    # W_h = tf.compat.v1.get_variable("W_h", [1, 1, attn_size, attn_size])
    # encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")

    W_h = nn.Parameter(th.empty((1, 1, attn_size, attn_size)))
    th.nn.init.xavier_uniform_(W_h)  # 使用Xavier初始化方法初始化
    encoder_features = F.conv2d(encoder_states, W_h, stride=1, padding=0)

    # v = tf.compat.v1.get_variable("v", [attn_size])
    v = nn.Parameter(th.empty((attn_size,)))
    nn.init.xavier_uniform_(v)  # 使用Xavier初始化方法初始化



    def attention(decoder_state):
        """Calculate the context vector and attention distribution from the decoder state."""

        # 使用卷积层对decoder_state进行卷积操作，得到decoder_features
        decoder_features = concat_conv([decoder_state], attn_size)

        # 在decoder_features上插入两个维度，以便与encoder_features形状匹配
        decoder_features = decoder_features.unsqueeze(1).unsqueeze(1)
        # 计算加权和，得到e向量，然后加上attn_mask
        e = th.sum(v * th.tanh(encoder_features + decoder_features), [2, 3])
        e += attn_mask
        # 对e向量进行softmax操作，得到注意力分布
        attn_dist = F.softmax(e, dim=1)
        # 计算上下文向量，通过对encoder_states进行加权求和
        context_vector = th.sum(th.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [2, 3])

        # 调整上下文向量的形状
        context_vector = context_vector.reshape(-1, attn_size)

        return context_vector, attn_dist


    outputs, attn_dists = [], []
    state = initial_state
    context_vector = th.zeros([batch_size, attn_size])
    print('context_vector.shape:', context_vector.shape)
    context_vector = context_vector.reshape(-1, attn_size)
    print('context_vector.shape:', context_vector.shape)
    # if initial_state_attention:
    #     context_vector, _ = attention(initial_state)

    # decoder_inputs = th.unstack(decoder_inputs, axis=1)
    decoder_inputs = th.split(decoder_inputs, 1, dim=1)#返回是元组
    decoder_inputs = th.squeeze(decoder_inputs, dim=1).tolist()
    print('INFO: Adding attention_decoder of {} timesteps...'.format(len(decoder_inputs)))
    len1 = decoder_inputs[0].shape[1] + context_vector.shape[1]
    len2 = 0
    input_size = decoder_inputs[0].shape[2]
    matrix1 = th.nn.linear((len1,))
    matrix2 = None
    bias1 = th.nn.linear(())
    bias2 = None

    # with th.no_grad():
    for i, inp in enumerate(decoder_inputs):
        # if i > 0:
        #     tf.compat.v1.get_variable_scope().reuse_variables()

        input_size = inp.shape[2]
        if input_size.value is None:
            raise ValueError("Could not infer input size from input: %s" % inp.name)
        
        input_size = inp.get_shape().with_rank(2)[1]
        
        x = concat_conv([inp] + [context_vector],matrix1, bias1)

        cell_output, state = cell(x, state)

        # if i == 0 and initial_state_attention:
        #     with tf.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
        #         context_vector, attn_dist = attention(state)
        # else:
        context_vector, attn_dist = attention(state)
        attn_dists.append(attn_dist)

        if not matrix2:
            len2 =   cell_output.shape[1] + context_vector.sahpe[1]
            matrix2 = nn.linear(len2,cell.output_size)
            bias2 = nn.linear(cell.output_size)
        output = concat_conv([cell_output] + [context_vector],matrix2,bias2)
        outputs.append(output)

    return outputs, state, attn_dists



def concat_conv(args,  matrix , bias , output_size ):
    if len(args) == 1:
        res = th.matmul(args[0], matrix)
    else:
        res = th.matmul(th.concat( args, dim=1), matrix)
    if any(bias):
        bias_variable = bias
        res += bias_variable 
    return res