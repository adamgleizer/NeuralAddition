# Neural Addition

A recurrent neural network that learns to add digits by posing it as a sequence-to-sequence translation problem.

Examples of this architecture in application include machine translation and text generation.

Created without the use of any deep learning libraries, only NumPy and AutoGrad.

# Using the file

To train a model, instantiate a parameter dictionary such as:

params = {'encoder': random_init(IN_SIZE, HIDDEN_SIZE), 'decoder': random_init(IN_SIZE, HIDDEN_SIZE)}.

Then simply call the training method, i.e:

weights = train(params).

At the end of training, instantiate an encoder and decoder with the trained weights and you're set.

Thanks!

