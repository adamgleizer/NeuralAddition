# -*- coding: utf-8 -*-
"""
@author: Adam Gleizer
"""
import autograd.numpy as np
from autograd import grad

np.random.seed(0)
vocab = sorted(set('0123456789+ ')) #space included for padding so all ints are 'equal length',
#eliminates the need for start and end of sequence tokens

class OneHot(object):
    
    def __init__(self):
        #construct a mapping from alphabet to one-hot position and it's inverse
        self.char_to_index = dict((c,i) for i, c in enumerate(vocab))
        self.index_to_char = dict((i, c) for i, c in enumerate(vocab))
        
    def encode(self, equation, max_len): 
        #transposed one-hot representation
        rep = np.zeros((max_len, len(vocab)))
        for i, c in enumerate(equation):
            rep[i, self.char_to_index[c]] = 1
        return rep
    
    def decode(self, out): 
        #decodes a matrix of probabilities back into characters
        max_indices = np.argmax(out, axis=-1)
        return ''.join(self.index_to_char[indices] for indices in max_indices)

#Model and dataset parameters
TRAINING_SIZE = 50000
DIGITS = 3
IN_SIZE = len(vocab)
HIDDEN_SIZE = 128
OUT_SIZE = len(vocab)
BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.01
#maximum length of input 'integer+integer', integer length is DIGITS
MAXLEN = 2*DIGITS + 1

table = OneHot()

questions = [] #addition equations
expected = [] #expected answers to questions
seen = set() #keep track of equations already seen to avoid duplicate data

print('Generating Data...')

while len(questions) < TRAINING_SIZE: #generate data
    f = lambda: int(''.join(np.random.choice(list('0123456789')) \
                                             for i in range(np.random.randint(1, DIGITS+1))))
    a, b = f(), f() 
    #Skipping any questions we've seen
    #Sorting to avoid having both x1+x2 and x2+x1 in the dataset
    check = tuple(sorted((a, b)))
    if check in seen:
        continue
    seen.add(check)
    #padding data with spaces so that all sequences are equal length
    equation = '{}+{}'.format(a,b)
    padded = equation + (' ' * (MAXLEN - len(equation)))
    answer = str(a+b)
    answer += ' ' * (DIGITS + 1 - len(answer))
    #Reversing the question to help learn long-term dependencies when passed through the encoder
    formatted = padded[::-1]
    questions.append(formatted)
    expected.append(answer)

#Vectorizing dataset
x = np.zeros((len(questions), MAXLEN, IN_SIZE))
y = np.zeros((len(questions), DIGITS + 1, OUT_SIZE))

for i, sentence in enumerate(questions):
    x[i] = table.encode(sentence, MAXLEN)
for i, answer in enumerate(expected):
    y[i] = table.encode(answer, DIGITS + 1)

#Data isn't truly 'random' as the generation of latter questions were dependent on previously generated
#questions due to the 'seen' check. Shuffling to help make difficulty uniform
indices = np.arange(len(y))
np.random.shuffle(indices)
x, y = x[indices], y[indices]

#Doing a 90/10 split of the data for training/testing respectively
split = len(x) - (len(x) // 10)
x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

#Seperating into batches
NUM_BATCHES = len(x_train) // BATCH_SIZE
x_train = np.reshape(x_train, (NUM_BATCHES, BATCH_SIZE, MAXLEN, IN_SIZE))
y_train = np.reshape(y_train, (NUM_BATCHES, BATCH_SIZE, DIGITS + 1, OUT_SIZE))
print('Data generated \n')
print('(num_batches, batch_size, seq_len, vocab_size) = ' + str(x_train.shape))

def sigmoid(x):
        return 1/(1+ np.exp(-x))
    
def softmax(X, theta = 1.0, axis = None): #This portion isn't mine, credit to Nolan Conaway for the softmax function
    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p
    
def random_init(input_size=IN_SIZE, hidden_size=HIDDEN_SIZE): #Xavier initialization of the parameters
    return {'Wiz': np.random.normal(size=(input_size, hidden_size))*(np.sqrt(1/(input_size + hidden_size))),
            'Wir': np.random.normal(size=(input_size, hidden_size))*(np.sqrt(1/(input_size + hidden_size))),
            'Win': np.random.normal(size=(input_size, hidden_size))*(np.sqrt(1/(input_size + hidden_size))),
            'Whz': np.random.normal(size=(hidden_size, hidden_size))*(np.sqrt(1/(2*hidden_size))),
            'Whr': np.random.normal(size=(hidden_size, hidden_size))*(np.sqrt(1/(2*hidden_size))),
            'Whn': np.random.normal(size=(hidden_size, hidden_size))*(np.sqrt(1/(2*hidden_size))),
            'bz': np.random.normal(size=hidden_size),
            'br': np.random.normal(size=hidden_size),
            'bg': np.random.normal(size=hidden_size),
            'out': np.random.normal(size=(hidden_size, OUT_SIZE))}

#Basic GRU cell for the encoder/decoder
class GRUCell(object):
    def __init__(self, params=random_init(IN_SIZE, HIDDEN_SIZE)):
        self.params = params
        
    def forward(self, current, h_prev):
        z_in = np.matmul(current, self.params['Wiz']) + np.matmul(h_prev, self.params['Whz']) + self.params['bz']
        z = sigmoid(z_in)
        r_in = np.matmul(current, self.params['Wir']) + np.matmul(h_prev, self.params['Whr']) + self.params['br']
        r = sigmoid(r_in)
        g_in = np.matmul(current, self.params['Win']) + np.multiply(np.matmul(h_prev, self.params['Whn']), r) + self.params['bg']
        g = np.tanh(g_in)
        h_current = np.multiply((1-z), g) + np.multiply(z, h_prev)
        return h_current

#Encoder portion of the network
class GRUEncoder(object):
    def __init__(self, params=None):
        if params == None:
            self.gru = GRUCell()
        else:
            self.gru = GRUCell(params)
        
    def forward(self, inputs): #Input is BS x Seq_len x vocab_size
        hidden = np.zeros((BATCH_SIZE, HIDDEN_SIZE)) #initial hidden states (none)
        
        for i in range(MAXLEN):
            t = inputs[:,i,:] #Current time step
            hidden = self.gru.forward(t, hidden)
        return hidden
    
#Decoder portion of the network
class RNNDecoder():
    def __init__(self, params=None):
        if params == None:
            self.gru = GRUCell()
        else:
            self.gru = GRUCell(params)
        
    def forward(self, inputs, hidden): 
        #initial hidden state comes from the encoder
        #inputs fed from answer during training (curriculum learning), feeds-back during generation
        input_init = np.zeros((BATCH_SIZE, IN_SIZE))
        hidden_t = self.gru.forward(input_init, hidden)
        outs = np.reshape(softmax(np.matmul(hidden_t, self.gru.params['out'])), (100,1,12))
        for i in range(1, DIGITS + 1):
            t = inputs[:,i,:]
            hidden_t = self.gru.forward(t, hidden_t)
            output = np.matmul(hidden_t, self.gru.params['out'])
            soft_max = np.reshape(softmax(output, axis=-1), (100,1,12))
            outs = np.concatenate((outs, soft_max), axis=1)
        return -np.log(outs)
    
def cross_entropy_loss(target, predictions): #averages the loss over each time step, then over the whole batch
    transposed = np.transpose(target, axes=(0,2,1))
    prod = np.matmul(predictions, transposed)
    return np.mean(np.mean(np.diagonal(prod, axis1=-1, axis2=-2), axis=1))  

class loss(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __call__(self, params_dict):
            encoder = GRUEncoder(params_dict['encoder'])
            decoder = RNNDecoder(params_dict['decoder'])
            predictions = decoder.forward(self.y, encoder.forward(self.x))
            return cross_entropy_loss(self.y, predictions)

def gd_step(cost, params_dict, lrate):
    costgrad = grad(cost)
    def update(A, B): 
        #helper function to speed up runtime 
        updated = {key: A[key] - B[key]*lrate for key in A.keys() if key in B.keys()}
        return updated
    grad_dict = costgrad(params_dict)
    for key in params_dict:
        params_dict[key] = update(params_dict[key], grad_dict[key])
    return params_dict

def train(params_dict):
    for i in range(EPOCHS):
        if i % 10 == 0:
            print('epoch number ' + str(i))
        for j in range(NUM_BATCHES):
            cost = loss(x_train[j], y_train[j])
            params_dict = gd_step(cost, params_dict, LEARNING_RATE)
        batch_num = np.random.randint(0, 450)
        sample_num = np.random.randint(0, 100)
        print('Question: ' + str(table.decode(x_train[batch_num,sample_num][::-1])).strip())
        encoder, decoder = GRUEncoder(params_dict['encoder']), RNNDecoder(params_dict['decoder'])
        sample = decoder.forward(y_train[batch_num], encoder.forward(x_train[batch_num]))
        prediction = table.decode(sample[sample_num])
        print('Prediction: ' + str(prediction))
        print('Current loss: ' + str(cost(params_dict)))
    return params_dict
