'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

import argparse


class InputConfig():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
        parser.add_argument('--train', type=bool, default=True)
        parser.add_argument('--show_predictions', default=False, type=bool, help='show predictions in the test stage')
        parser.add_argument('--word_vector', default='./glove/glove.6B.100d.txt', help='word vector')
        parser.add_argument('--prefix', default='dev', help='prefix for storing model and log')
        parser.add_argument('--num_class', default=2, type=int, help='number of classes')
        parser.add_argument('--freeze_bert', action='store_true', default=False, help='freeze parameters of bert encoder')
        parser.add_argument('--single', action='store_true', default=False, help='single string or multiple strs for bert encoders')
        parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
        parser.add_argument('--delta_epoch', default=0, type=int, help='number of additional epochs')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
        parser.add_argument('--base_lr', default=0.000001, type=float, help='learning rate')
        parser.add_argument('--l2', default=0.00001, type=float, help='weight decay')
        parser.add_argument('--learning_rate_decay', default=0.98, type=float, help='learning rate decay factor')

        # model parameter
        parser.add_argument('--n_rels', default=16, type=int, help='number of relations')
        parser.add_argument('--n_rels_emo', default=6, type=int, help='number of emotional relations')
        parser.add_argument('--plm_type', default='roberta', help='PLM encoder used in the model')
        parser.add_argument('--n_encoder_hidden', default=768, type=int, help='dimension of encoder hideen layer')
        parser.add_argument('--encoder_dropout', default=0.33, type=float, help='dropout of encoder')
        parser.add_argument('--n_arc_mlp', default=2000, type=int, help='output numbers of arc_mlp layer')
        parser.add_argument('--n_rel_mlp', default=2000, type=int, help='output numbers of rel_mlp layer')
        parser.add_argument('--mlp_dropout', default=0.33, type=float, help='dropout of mlp layer')
        parser.add_argument('--scale', default=0, type=int, help='scale used in the arc_attn')
        parser.add_argument('--alpha',  nargs='+', required=True, help='regularizer of loss') #default [1.0, 1.0, 1.0, 1.0, 0.05, 0.05]
        parser.add_argument('--multi_par', action='store_true', default=True, help='whether a dependent has multiple heads')
        parser.add_argument('--supervision_loss', default='kl', type=str, help='type of supervision loss')
        parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker information')
        parser.add_argument('--use_turn', action='store_true', default=False, help='whether to use turn information')
        parser.add_argument('--max_norm', default=10, type=int, help='max norm of gradient clip')
        parser.add_argument('--warm_up_ratio', default=0.1, type=float, help='warm up ratio:wq:')
        parser.add_argument('--layer_num', default=12, type=int, help='number of layer norm in BERT')
        parser.add_argument('--tune_start_layer', default=9, type=int, help='number of layer to start tune')
        parser.add_argument('--dropout_emb', default=0.3, type=float, help='dropout rate of encoderhead')
        parser.add_argument('--shared_mlp', action='store_true', default=False, help='whether to share mlp')
        parser.add_argument('--saved_model_path', default=r'./saved_model/', type=str, help='path of saved model')
        self.args = parser.parse_args()