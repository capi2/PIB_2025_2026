import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def get_matrix_results(matrix_files):
    matrices = []
    for matrix_file in matrix_files:
        aggregated_matrice = np.zeros((3,3), dtype=int)

        with open(matrix_file, "r") as matrixfile:
            lines = matrixfile.readlines()
            m = []
            for line in lines:
                if line == "\n":
                    aggregated_matrice += np.array(m)
                    m = []
                else:
                    line = line.strip("[]").split()
                    line = np.array([x.strip("]") for x in line], dtype=int)
                    m.append(line)

        matrices.append(aggregated_matrice)
    return matrices

adab =  "results_matrices/AdaBoostClassifierconfusionmatricescomplete"
gbm =   "results_matrices/GradientBoostingClassifierconfusionmatricescomplete"
rf =    "results_matrices/RandomForestClassifierconfusionmatricescomplete"
xgb =   "results_matrices/XGBClassifierconfusionmatricescomplete"

titles = ["Adaboost", "Gradient Boosting", "Random Forest", "eXtreme Gradient Boosting"]

matrix_files = [adab, gbm, rf, xgb]
matrices = get_matrix_results(matrix_files)

fig, axes = plt.subplots(2, 2, figsize = (15,10))
m = 0
for i in range(2):
    for j in range(2):
        disp = ConfusionMatrixDisplay(matrices[m], 
                                      display_labels=["Chikungunya", "Dengue", "Discarded/ND"])
        axes[i][j].set_title(titles[m])
        disp.plot(ax=axes[i][j], values_format='d', cmap="Blues")
        
        disp.ax_.set_xlabel("")
        disp.ax_.set_ylabel("")
        m += 1

fig.suptitle("Confusion Matrices for Adab, XGB, GBM and RF")
plt.savefig("crossvalidationmatrices.png", format='png')
plt.show()