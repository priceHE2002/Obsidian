```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 画图逻辑 (随便画点什么)
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Default Settings")

# 2. 只需要这一行！
save_plot("sin_wave_test")
```
