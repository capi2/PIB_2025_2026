import pandas as pd

print("RESULTADOS CLINICOS MACRO")

# TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results_clinical/AdaBoostClassifierclinical.csv"
gnb =   "results_clinical/GaussianNBclinical.csv"
gbc =   "results_clinical/GradientBoostingClassifierclinical.csv"
knn =   "results_clinical/KNeighborsClassifierclinical.csv"
mlp =   "results_clinical/MLPClassifierclinical.csv"
rf =    "results_clinical/RandomForestClassifierclinical.csv"
xgb =   "results_clinical/XGBClassifierclinical.csv"

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

adab =  "results_clinical/AdaBoostClassifierchikdengclinical.csv"
gnb =   "results_clinical/GaussianNBchikdengclinical.csv"
gbc =   "results_clinical/GradientBoostingClassifierchikdengclinical.csv"
knn =   "results_clinical/KNeighborsClassifierchikdengclinical.csv"
mlp =   "results_clinical/MLPClassifierchikdengclinical.csv"
rf =    "results_clinical/RandomForestClassifierchikdengclinical.csv"
xgb =   "results_clinical/XGBClassifierchikdengclinical.csv"

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

adab =  "results_clinical/AdaBoostClassifierdengundefclinical.csv"
gnb =   "results_clinical/GaussianNBdengundefclinical.csv"
gbc =   "results_clinical/GradientBoostingClassifierdengundefclinical.csv"
knn =   "results_clinical/KNeighborsClassifierdengundefclinical.csv"
mlp =   "results_clinical/MLPClassifierdengundefclinical.csv"
rf =    "results_clinical/RandomForestClassifierdengundefclinical.csv"
xgb =   "results_clinical/XGBClassifierdengundefclinical.csv"

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

adab =  "results_clinical/AdaBoostClassifierchikundefclinical.csv"
gnb =   "results_clinical/GaussianNBchikundefclinical.csv"
gbc =   "results_clinical/GradientBoostingClassifierchikundefclinical.csv"
knn =   "results_clinical/KNeighborsClassifierchikundefclinical.csv"
mlp =   "results_clinical/MLPClassifierchikundefclinical.csv"
rf =    "results_clinical/RandomForestClassifierchikundefclinical.csv"
xgb =   "results_clinical/XGBClassifierchikundefclinical.csv"

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