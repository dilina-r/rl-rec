

from collections import OrderedDict
gru4rec_params = OrderedDict([('layers', [96]), 
                                ('batch_size', 144), 
                                ('learning_rate', 0.05), 
                                ('dropout_p_embed', 0.2), 
                                ('dropout_p_hidden', 0.25), 
                                ('momentum', 0.55), 
                                ('sample_alpha',0.6), 
                                ('bpreg', 0.45), 
                                ('elu_param', 0), 
                                ('constrained_embedding', True), 
                                ('loss', 'cross-entropy'), 
                                ('embedding', 0), 
                                ('n_epochs', 20), 
                                ('n_sample', 2048), 
                                ('logq', 1.0)])
