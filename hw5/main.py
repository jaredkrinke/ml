import numpy as np
import code_for_hw5 as hw5
from auto import auto_data, auto_values, sigma

for i, data in enumerate(auto_data):
    for order in range(1, 4):
        transform = hw5.make_polynomial_feature_fun(order)
        for lam in np.arange(0, 0.11, 0.01) if order <= 2 else np.arange(0, 220, 20):
            transformed_data = transform(data)
            e = (hw5.xval_learning_alg(transformed_data, auto_values, lam, 10)[0][0]) * sigma[0]
            print(f"{'{0:0.4f}'.format(e)}\tFeature set {i + 1}\tOrder {order}\tLambda {lam}")
