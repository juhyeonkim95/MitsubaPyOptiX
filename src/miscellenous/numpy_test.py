import numpy as np

a = np.zeros((3, ))
print(np.prod(a.shape))
print(np.prod((3,)))

a = np.zeros((4096, 256))
b = np.zeros((256, 256))
c = np.einsum('ij,jk->ijk', a, b)
print(c.shape)
