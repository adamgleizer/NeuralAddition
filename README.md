# Neural Addition

A recurrent neural network that learns to add digits by posing it as a sequence-to-sequence translation problem.

Examples of this architecture in application include machine translation and text generation.

Created without the use of any deep learning libraries, only NumPy and Autograd.

# Using the file

To train a model, instantiate a parameter dictionary like so:

params = {'encoder': random_init(), 'decoder': random_init()}.

Then simply call the training method, i.e:

weights = train(params).

At the end of training, instantiate an encoder and decoder with the trained weights and you're set.

Thanks!

