import numpy as np
import math as m

class Module:
    def step(self, lrate): pass  # For modules w/o weights


# Linear modules
#
# Each linear module has a forward method that takes in a batch of
# activations A (from the previous layer) and returns
# a batch of pre-activations Z.
#
# Each linear module has a backward method that takes in dLdZ and
# returns dLdA. This module also computes and stores dLdW and dLdW0,
# the gradients with respect to the weights.
class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        return self.W.T @ A + self.W0  # Your code (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW  = self.A @ dLdZ.T  # Your code
        self.dLdW0 = np.sum(dLdZ, axis=1, keepdims=True)  # Your code
        return self.W @ dLdZ        # Your code: return dLdA (m x b)

    def step(self, lrate):  # Gradient descent step
        self.W  -= lrate * self.dLdW  # Your code
        self.W0  -= lrate * self.dLdW0  # Your code


# Activation modules
#
# Each activation module has a forward method that takes in a batch of
# pre-activations Z and returns a batch of activations A.
#
# Each activation module has a backward method that takes in dLdA and
# returns dLdZ, with the exception of SoftMax, where we assume dLdZ is
# passed in.
class Tanh(Module):  # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z) # nxb
        return self.A

    def backward(self, dLdA):  # Uses stored self.A # nxb
        dAdZ = 1 - (self.A * self.A)
        return dAdZ * dLdA  # Your code: return dLdZ (n, b)


class ReLU(Module):  # Layer activation
    def forward(self, Z): # nx1
        self.A = np.maximum(0, Z)  # Your code: (?, b) nxb
        return self.A

    def backward(self, dLdA):  # uses stored self.A
        return np.where(self.A > 0, dLdA, 0) # Your code: return dLdZ (?, b)


class SoftMax(Module):  # Output activation
    def forward(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)  # Your code: (?, b)

    def backward(self, dLdZ):  # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred, axis=0)  # Your code: (1, b)


# Loss modules
#
# Each loss module has a forward method that takes in a batch of
# predictions Ypred (from the previous layer) and labels Y and returns
# a scalar loss value.
#
# The NLL module has a backward method that returns dLdZ, the gradient
# with respect to the preactivation to SoftMax (note: not the
# activation!), since we are always pairing SoftMax activation with
# NLL loss
class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred # nxb
        self.Y = Y # nxb
        return -np.sum(Y * np.log(Ypred))  # Your code: return loss (scalar)

    def backward(self):  # Use stored self.Ypred, self.Y
        return self.Ypred - self.Y # Your code (?, b) nxb

# Batch normalization
class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(A, axis=1, keepdims=True)  # Your Code
        self.vars = np.var(A, axis=1, keepdims=True)  # Your Code

        # Normalize inputs using their mean and standard deviation
        self.norm = (A - self.mus) / (np.sqrt(self.vars) + self.eps)  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.G * self.norm + self.B  # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B  -= lrate * self.dLdB  # Your code
        self.G  -= lrate * self.dLdG  # Your code
        return

# Neural Network implementation
class Sequential:
    def __init__(self, modules, loss):  # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        for it in range(iters):
            index = np.random.randint(Y.shape[1])
            Xt = X[:,index:index + 1]
            Yt = Y[:,index:index + 1]
            Ypred = self.forward(Xt)
            loss = self.loss.forward(Ypred, Yt)
            # TODO: Is the loss used anywhere? Or only implicitly in backward?
            dLdA = self.loss.backward()
            self.backward(dLdA)
            self.step(lrate)

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:,indices] # Your code
            Y = Y[:,indices] # Your code

            index = 0
            for j in range(m.floor(N/K)):
                if num_updates >= iters: break

                # Your code
                Xt = X[:,index:index + K]
                Yt = Y[:,index:index + K]
                index += K
                Ypred = self.forward(Xt)
                self.loss.forward(Ypred, Yt)
                dLdA = self.loss.backward()
                self.backward(dLdA)
                self.step(lrate)
                
                num_updates += 1

    def forward(self, Xt):  # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):  # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]:
            # Note that delta can refer to dLdA or dLdZ over the
            # course of the for loop, depending on the module m
            delta = m.backward(delta)

    def step(self, lrate):  # Gradient descent step
        for m in self.modules: m.step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '\tAcc =', acc, '\tLoss =', cur_loss, flush=True)
