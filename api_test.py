import mindspore.nn as nn
from mindspore import Tensor
import mindspore
from mindspore import context



from queue import Queue


q = Queue(maxsize=5)
print(q.full())
q.put(1)
q.put(1)
q.put(1)
q.put(1)
q.put(1)
print(q.full())