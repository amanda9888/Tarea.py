checkout main
git checkout -b ejercicio2
git push origin ejercicio2


tp, tn, fp, fn = 40, 30, 20, 10

accuracy = (tp + tn) / (tp + tn + fp + fn)  # 0.7
precision = tp / (tp + fp)                   # 0.6667
recall = tp / (tp + fn)                      # 0.8
f1 = 2 * (precision * recall) / (precision + recall)  # 0.7273
