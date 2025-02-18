

from collections import OrderedDict


gru4rec_params = OrderedDict([('layers', [96]), 
                              ('batch_size', 144), 
                              ('learning_rate', 0.14), 
                              ('dropout_p_embed', 0.45), 
                              ('dropout_p_hidden', 0.25), 
                              ('momentum', 0.5), 
                              ('sample_alpha', 0.1), 
                              ('bpreg', 1.75), 
                              ('elu_param', 0.5), 
                              ('constrained_embedding', True), 
                              ('loss', 'cross-entropy'), 
                              ('embedding', 0), 
                              ('n_epochs', 20), 
                              ('n_sample', 2048), 
                              ('logq', 1.0)])



# gru4rec_ori = OrderedDict([
#     ('loss', 'cross-entropy'),
#     ('constrained_embedding', True),
#     ('embedding', 0),
#     ('elu_param', 0),
#     ('layers', [512]),
#     ('n_epochs', 10),
#     ('batch_size', 240),
#     ('dropout_p_embed', 0.45),
#     ('dropout_p_hidden', 0.0),
#     ('learning_rate', 0.065),
#     ('momentum', 0.0),
#     ('n_sample', 2048),
#     ('sample_alpha', 0.5),
#     ('bpreg', 0.0),
#     ('logq', 1.0)
# ])


# for key, value in gru4rec_ori.items():
#     if key not in gru4rec_params:
#         print(key)
#         gru4rec_params[key] = value

# print(gru4rec_params)
