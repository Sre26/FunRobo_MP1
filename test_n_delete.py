# test
import numpy as np

arr = np.array([1, 0, 0, 2, 3, 1, 0])
log_ans = (arr > 2) + (arr < 1)

print(~log_ans)

log_ans2 = ~((arr > 2) + (arr < 1))
#print(~(arr < 1))
print(log_ans2)

t2 = np.ones([7]).astype(bool)
x = np.logical_and(t2,log_ans2)
print(x)