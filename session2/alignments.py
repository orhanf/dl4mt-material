'''
Extracts alignments of a translation file give its source, and saves
the alignments for each translation to a file.
'''
import argparse

import cPickle as pkl
import numpy
import os
import matplotlib
matplotlib.use('Agg')
import pylab
import theano

from nmt import (build_model, load_params,
                 init_params, init_tparams)

floatX = theano.config.floatX


# builds alingment computational graph and compiles its function
def build_alignment_cg(model, options):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build model
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, options)
    inps = [x, x_mask, y, y_mask]

    # compile a function and return it
    return theano.function(inps, opt_ret['dec_alphas'])


# wrapper function to call f_align
def get_alignments(f_align, x, y):
    return f_align(x, numpy.ones_like(x).astype(floatX),
                   y, numpy.ones_like(y).astype(floatX))


def main(model, dictionary, dictionary_target, source_file, target_file,
         chr_level=False, saveto='./'):

    # load model model_options
    print 'Loading model options'
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    print 'Loading dictionaries'
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # utility function
    def _seq2words(seq, idict):
        ww = []
        for w in seq:
            if w == 0:
                break
            ww.append(idict[w])
        return ww

    def _words2seq(words, wdict):
        if chr_level:
            words = list(words.decode('utf-8').strip())
        else:
            words = words.strip().split()
        seq = map(lambda w: wdict[w] if w in wdict else 1, words)
        seq = map(lambda ii: ii if ii < options['n_words'] else 1, seq)
        seq += [0]
        return seq

    # get alignment function
    print 'Building model'
    f_align = build_alignment_cg(model, options)

    print 'Extracting alingments'
    with open(source_file, 'r') as src:
        with open(target_file, 'r') as trg:
            for idx, (x_, y_) in enumerate(zip(src, trg)):
                if idx % 100 == 0 and idx != 0:
                    print '...processed {} lines'.format(idx)

                # get alignments
                x = numpy.array(_words2seq(x_, word_dict))
                y = numpy.array(_words2seq(y_, word_dict_trg))
                alignments = get_alignments(f_align, x[:, None], y[:, None])

                # re arrange them for plotting
                alignments = numpy.asarray(alignments)
                alignments = alignments.reshape(alignments.shape[0],
                                                alignments.shape[2])
                # plot and save
                pylab.clf()
                pylab.tick_params(axis='both', which='major', labelsize=5)
                pylab.tick_params(axis='both', which='minor', labelsize=5)

                pylab.yticks([i for i in range(len(y))],
                             [j.decode('utf-8')
                              for j in _seq2words(y, word_idict_trg)])
                pylab.xticks([i for i in range(len(x))],
                             [j.decode('utf-8')
                              for j in _seq2words(x, word_idict)])
                pylab.gray()
                pylab.imshow(alignments, interpolation="none")
                pylab.savefig(
                    os.path.join(saveto, "alignments_%d.png" % idx))

    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.target, chr_level=args.c, saveto=args.saveto)
