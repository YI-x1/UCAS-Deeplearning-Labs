import matplotlib.pyplot as plt


epochs = [1, 2, 3, 4, 5]
avg_train_loss = [0.152, 0.051, 0.032, 0.042, 0.026]  # 计算得到的平均损失
test_accuracy = [98.36, 98.36, 98.83, 98.79, 99.05]  # 测试准确率

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, avg_train_loss, 'bo-', label='Train Loss')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracy, 'ro-', label='Test Accuracy')
plt.xlabel('Epoch'), plt.ylabel('Accuracy (%)'), plt.title('Test Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()