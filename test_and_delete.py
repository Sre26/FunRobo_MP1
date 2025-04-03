# test
import numpy as np

test = np.array([1, 0, 3, 0, 0, 2])
print(test)

logic_answer1 = test < 1
print(logic_answer1)
logic_answer2 = test > 2.5
print(logic_answer2)

test2 = logic_answer1 + logic_answer2
print(~test2)

logic_answer3 = int(logic_answer1) + int(logic_answer2)
print(logic_answer3)

logic_answer = logic_answer3 < 1
print(logic_answer)