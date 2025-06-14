"""
姓名：李碧萱
学号:2023152605

本代码为第五次作业第二题的实现，题目要求：利用大模型，参照“二分类.txt”中的 Python 代码，
设计一个具有实际应用意义的二分类案例，并使用感知器算法实现其分类功能。

本示例基于感知器模型，通过三个输入特征 —— 收缩压（mmHg）、年龄和性别，来判断一个人的血压状态是否正常。
性别被编码为数字形式：0 表示女性，1 表示男性。

模型输出为二分类标签：
- 1.0 表示血压在合理范围内，属于“正常”；
- 0.0 表示血压异常（可能偏高或偏低）。

"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta=0.01, iterations=10):
        self.lr = eta
        self.iterations = iterations
        self.w = None  # 初始化为向量
        self.bias = 0.0

    def fit(self, X, Y):
        self.w = np.zeros(X.shape[1])  # X 是二维矩阵，每行一个样本
        for _ in range(self.iterations):
            for xi, yi in zip(X, Y):
                update = self.lr * (yi - self.predict(xi))
                self.w += update * xi
                self.bias += update

    def net_input(self, x):
        return np.dot(x, self.w) + self.bias  # 点积 + 偏置

    def predict(self, x):
        return 1.0 if self.net_input(x) > 0.0 else 0.0

# 示例数据：每条数据 [血压, 年龄, 性别]
# 性别：0 = 女，1 = 男
X_train = np.array([
    [85, 70, 1],     # 高龄男性，低血压 → 异常
    [120, 25, 1],    # 青年男性，正常血压 → 正常
    [140, 60, 0],    # 中老年女性，边界血压 → 正常
    [150, 50, 1],    # 中年男性，高血压 → 异常
    [95, 80, 0],     # 老年女性，稍低 → 异常
    [110, 35, 0],    # 成年女性，正常 → 正常
    [160, 45, 1],    # 高血压 → 异常
    [118, 28, 1],    # 青年男性 → 正常
])

# 标签：1 = 正常，0 = 异常
Y_train = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

# 训练模型
model = Perceptron(eta=0.01, iterations=10)
model.fit(X_train, Y_train)

# 测试样本
X_test = np.array([
    [130, 40, 1],   # 正常血压中年男
    [88, 75, 0],    # 老年女，偏低
    [145, 60, 1],   # 高压老年男
    [115, 32, 0],   # 青年女性
])

print("\n测试预测结果：")
for person in X_test:
    result = model.predict(person)
    status = "正常" if result == 1.0 else "异常"
    print(f"输入（血压:{person[0]}，年龄:{person[1]}，性别:{'男' if person[2] == 1 else '女'}）=> {status}")

print("\n模型参数:")
print("权重 w =", model.w)
print("偏置 b =", model.bias)
