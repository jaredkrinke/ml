from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        self.state = self.start_state
        output = []
        for x in input_seq:
            self.state = self.transition_fn(self.state, x)
            output.append(self.output_fn(self.state))
        return output


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = [0, 0]

    def transition_fn(self, s, x):
        a, b = x
        carry = s[1]
        return [a ^ b ^ carry, (a & b) | (a & carry) | (b & carry)]

    def output_fn(self, s):
        return s[0]


class Reverser(SM):
    # TODO: Rewrite using pairs and list concatenation
    start_state = {"reading": True, "sequence": [], "output": None}

    def transition_fn(self, s, x):
        reading = True
        sequence = s["sequence"].copy()
        output = None
        if x == "end" or not s["reading"]:
            reading = False
            if len(sequence) > 0:
                output = sequence.pop()
        else:
            sequence.append(x)
        return {"reading": reading, "sequence": sequence, "output": output}

    def output_fn(self, s):
        return s["output"]


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.start_state = np.zeros([Wss.shape[0],1])
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2

    def transition_fn(self, s, i):
        return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, i) + self.Wss_0)

    def output_fn(self, s):
        return self.f2(np.dot(self.Wo, s) + self.Wo_0)

Wsx = np.array([[1]])
Wss = np.identity(1)
Wo = np.identity(1)
Wss_0 = np.zeros([1, 1])
Wo_0 = np.zeros([1, 1])
f1 = lambda x: x.copy()
f2 = lambda x: np.sign(x)
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)
