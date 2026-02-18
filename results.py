import pandas as pd

adab = "AdaBoostClassifierdengchik.csv"
gnb = "GaussianNBdengchik.csv"
gbc = "GradientBoostingClassifierdengchik.csv"
knn = "KNeighborsClassifierdengchik.csv"
mlp = "MLPClassifierdengchik.csv"
rf = "RandomForestClassifierdengchik.csv"
xgb = "XGBClassifierdengchik.csv"

result_files = [adab, gnb, gbc, knn, mlp, rf, xgb]

tab1 = ""
tab2 = ""
for file in result_files:
    df = pd.read_csv(file)

    columns = ["accuracy", "precision", "recall", "f1"]
    df.columns = columns

    tab1 += str(file)
    tab2 += str(file)
    
    #media -> desvio padrao
    #print(f"results for {file}")
    for name in columns:
        med = df[name].mean()
        std = df[name].std()
        tab1 += "," + str(med)
        tab2 += "," + str(std)
    tab1 += "\n"
    tab2 += "\n"

print("Model,Accuracy,Precision,Recall,f1-score")
print(tab1)
print("desv pad")
print(tab2)
