

from collections import OrderedDict


gru4rec_params = OrderedDict([('layers', [96]), 
                              ('batch_size', 208), 
                              ('learning_rate', 0.14), 
                              ('dropout_p_embed', 0.45), 
                              ('dropout_p_hidden', 0.0), 
                              ('momentum', 0.5), 
                              ('sample_alpha', 0.2), 
                              ('bpreg', 0.9), 
                              ('elu_param', 0), 
                              ('constrained_embedding', True), 
                              ('loss', 'bpr-max'), 
                              ('embedding', 0), 
                              ('n_epochs', 20), 
                              ('n_sample', 2048), 
                              ('logq', 1.0)])
