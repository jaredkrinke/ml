import numpy as np

# Back propgation through time
# xs is matrix of inputs: l by k
# dLdz2 is matrix of output errors:  1 by k
# states is matrix of state values: m by k
def bptt(self, xs, dLtdz2, states):
    dWsx = np.zeros_like(self.Wsx)
    dWss = np.zeros_like(self.Wss)
    dWo = np.zeros_like(self.Wo)
    dWss0 = np.zeros_like(self.Wss0)
    dWo0 = np.zeros_like(self.Wo0)
    # Derivative of future loss (from t+1 forward) wrt state at time t
    # initially 0;  will pass "back" through iterations
    dFtdst = np.zeros((self.hidden_dim, 1))
    k = xs.shape[1]
    # Technically we are considering time steps 1..k, but we need
    # to index into our xs and states with indices 0..k-1
    for t in range(k-1, -1, -1):
        # Get relevant quantities
        xt = xs[:, t:t+1]
        st = states[:, t:t+1]
        stm1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
        dLtdz2t = dLtdz2[:, t:t+1]
        # Compute gradients step by step
        # ==> Use self.df1(st) to get dfdz1;
        # ==> Use self.Wo, self.Wss, etc. for weight matrices
        # derivative of loss at time t wrt state at time t
        dLtdst = np.transpose(self.Wo) @ dLtdz2t        # Your code
        # derivatives of loss from t forward
        dFtm1dst = dLtdst + dFtdst             # Your code
        dFtm1dz1t = self.df1(st) * dFtm1dst           # Your code
        dFtm1dstm1 = np.transpose(self.Wss) @ dFtm1dz1t           # Your code
        # gradients wrt weights
        dLtdWo = dLtdz2t @ np.transpose(st)              # Your code
        dLtdWo0 = dLtdz2t             # Your code
        dFtm1dWss = dFtm1dz1t @ np.transpose(stm1)           # Your code
        dFtm1dWss0 = dFtm1dz1t           # Your code
        dFtm1dWsx = dFtm1dz1t @ np.transpose(xt)           # Your code
        # Accumulate updates to weights
        dWsx += dFtm1dWsx
        dWss += dFtm1dWss
        dWss0 += dFtm1dWss0
        dWo += dLtdWo
        dWo0 += dLtdWo0
        # pass delta "back" to next iteration
        dFtdst = dFtm1dstm1
    return dWsx, dWss, dWo, dWss0, dWo0
