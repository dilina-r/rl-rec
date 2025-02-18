

from collections import OrderedDict
gru4rec_params = OrderedDict([('layers', [128]), 
                                ('batch_size', 224), 
                                ('learning_rate', 0.25), 
                                ('dropout_p_embed', 0.35), 
                                ('dropout_p_hidden', 0.4), 
                                ('momentum', 0.0), 
                                ('sample_alpha',0.3), 
                                ('bpreg', 0.6), 
                                ('elu_param', 0.5), 
                                ('constrained_embedding', True), 
                                ('loss', 'cross-entropy'), 
                                ('embedding', 0), 
                                ('n_epochs', 20), 
                                ('n_sample', 2048), 
                                ('logq', 1.0)])
