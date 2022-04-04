import numpy as np

experience = [(0, 'b', 0), #t = 0
              (2, 'b', 0),
              (3, 'b', 2),
              (0, 'b', 0), #t = 3
              (2, 'b', 0),
              (3, 'c', 2),
              (0, 'c', 0), #t = 6
              (1, 'b', 1),
              (0, 'b', 0),
              (2, 'c', 0), #t = 9
              (3, 'c', 2),
              (0, 'c', 0),
              (1, 'c', 1), #t = 12
              (0, 'c', 0),
              (2, 'b', 0),
              (3, 'b', 2), #t = 15
              (0, 'b', 0),
              (2, 'c', 0),
              (3, '', 0), #t = 18
              ]

experience = [(state, 0 if action == "b" else 1, reward) for state, action, reward in experience]

alpha = 0.5
gamma = 0.9
q = np.zeros([4, 2])
for index, (state, action, reward) in enumerate(experience):
    state_next = experience[index + 1][0]
    q[state, action] = (1 - alpha) * q[state, action] + alpha * (reward + gamma * np.max(q[state_next]))
    print(q[state, action])
