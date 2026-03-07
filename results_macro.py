import pandas as pd

# TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results/AdaBoostClassifier.csv"
gnb =   "results/GaussianNB.csv"
gbc =   "results/GradientBoostingClassifier.csv"
knn =   "results/KNeighborsClassifier.csv"
mlp =   "results/MLPClassifier.csv"
rf =    "results/RandomForestClassifier.csv"
xgb =   "results/XGBClassifier.csv"

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
print("Model,Accuracy,Precision,Recall,f1-score")
print(tab2)


#DENGUE X CHIKUNGUNYA

print("DENGUE X CHIKUNGUNYA")

adab =  "results/AdaBoostClassifierdengchik.csv"
gnb =   "results/GaussianNBdengchik.csv"
gbc =   "results/GradientBoostingClassifierdengchik.csv"
knn =   "results/KNeighborsClassifierdengchik.csv"
mlp =   "results/MLPClassifierdengchik.csv"
rf =    "results/RandomForestClassifierdengchik.csv"
xgb =   "results/XGBClassifierdengchik.csv"

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
print("Model,Accuracy,Precision,Recall,f1-score")
print(tab2)




#DENGUE X NAO DEFINIDO

print("DENGUE X NAO DEFINIDO")

adab =  "results/AdaBoostClassifierdengundef.csv"
gnb =   "results/GaussianNBdengundef.csv"
gbc =   "results/GradientBoostingClassifierdengundef.csv"
knn =   "results/KNeighborsClassifierdengundef.csv"
mlp =   "results/MLPClassifierdengundef.csv"
rf =    "results/RandomForestClassifierdengundef.csv"
xgb =   "results/XGBClassifierdengundef.csv"

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
print("Model,Accuracy,Precision,Recall,f1-score")
print(tab2)


#CHIKUNGUNYA X NAO DEFINIDO

print("CHIKUNGUNYA X NAO DEFINIDO")

adab =  "results/AdaBoostClassifierchikundef.csv"
gnb =   "results/GaussianNBchikundef.csv"
gbc =   "results/GradientBoostingClassifierchikundef.csv"
knn =   "results/KNeighborsClassifierchikundef.csv"
mlp =   "results/MLPClassifierchikundef.csv"
rf =    "results/RandomForestClassifierchikundef.csv"
xgb =   "results/XGBClassifierchikundef.csv"

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
print("Model,Accuracy,Precision,Recall,f1-score")
print(tab2)