import pandas as pd

adab = "AdaBoostClassifierdengchik_perclass.csv"
gnb = "GaussianNBdengchik_perclass.csv"
gbc = "GradientBoostingClassifierdengchik_perclass.csv"
knn = "KNeighborsClassifierdengchik_perclass.csv"
mlp = "MLPClassifierdengchik_perclass.csv"
rf = "RandomForestClassifierdengchik_perclass.csv"
xgb = "XGBClassifierdengchik_perclass.csv"

result_files = [adab, gnb, gbc, knn, mlp, rf, xgb]

tab1 = ""
tab2 = ""

for file in result_files:
    df = pd.read_csv(file)

    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1"]
    df.columns = columns

    tab1 += str(file)
    tab2 += str(file)

    for name in columns:
        med = df[name].mean()
        std = df[name].std()

        tab1 += "," + str(med)
        tab2 += "," + str(std)
    tab1 += "\n"
    tab2 += "\n"
        
print("Model,Chik Precision,Chik Recall,Chik f1-score, Dengue Precision, Dengue Recall, Dengue f1-score")
print(tab1)
print("desv pad")
print(tab2)
