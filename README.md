# Neural Addition

A recurrent neural network that learns to add digits by posing it as a sequence-to-sequence translation problem.

Examples of this architecture in application include machine translation and text generation.

Created without the use of any deep learning libraries, only NumPy and Autograd.

Mostly an educational adventure. Large neural network libraries such as PyTorch and TensorFlow are very useful in terms of removing 'cognitive overhead', but doing everything yourself at least once helps form a firm foundation of what goes on 'behind the scenes'.
This leads to a far stronger ability to diagnose issues and problem-solve when using such libraries.

# Using the file

To train a model, instantiate a parameter dictionary like so:

params = {'encoder': random_init(), 'decoder': random_init()}.

Then simply call the training method, i.e:

weights = train(params).

At the end of training, instantiate an encoder and decoder with the trained weights. One hot-encode an addition question (as a string),
and feed forward.

Thanks!

