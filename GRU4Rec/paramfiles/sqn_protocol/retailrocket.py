

from collections import OrderedDict


gru4rec_params = OrderedDict([('layers', [384]), 
                              ('batch_size', 32), 
                              ('learning_rate', 0.035), 
                              ('dropout_p_embed', 0.4), 
                              ('dropout_p_hidden', 0.2), 
                              ('momentum', 0.1), 
                              ('sample_alpha', 0.4), 
                              ('bpreg', 0.05), 
                              ('elu_param', 0), 
                              ('constrained_embedding', True), 
                              ('loss', 'cross-entropy'), 
                              ('embedding', 0), 
                              ('n_epochs', 20), 
                              ('n_sample', 2048), 
                              ('logq', 1.0)])
