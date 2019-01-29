import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Reshape, 

class CFModel(Sequential):
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        U = Sequential()
        U = U.add(Embedding(n_users, k_factors, input_length=1))
        U = U.add(Reshape((k_factors,)))

        M = Sequential()
        M = M.add(Embedding(m_items, k_factors, input_length=1))
        M = M.add(Reshape((k_factors,)))

        super(CFModel, self).__init__(**kwargs)

        self.add(Dot([U, M]))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

