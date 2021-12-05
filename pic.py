import matplotlib.pyplot as plt
import numpy as np

loss_file = r"D:\Code\python\idiomBert\data\loss.txt"
score_file = r"D:\Code\python\idiomBert\data\score.txt"
losses = np.loadtxt(loss_file)
scores = np.loadtxt(score_file)

plt.figure(figsize = (7,5))
# plt.plot(x, y, 'g-', label=u'Dense_Unet(block layer=5)')
plt.plot(np.arrange(len(losses)), losses,'r-', label = u'RCSCA_Net')
plt.show()