import pandas as pd

print("RESULTADOS COM QUASE TODOS AS FEATURES MACRO")

# TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results_complete/AdaBoostClassifiercomplete.csv"
gnb =   "results_complete/GaussianNBcomplete.csv"
gbc =   "results_complete/GradientBoostingClassifiercomplete.csv"
knn =   "results_complete/KNeighborsClassifiercomplete.csv"
mlp =   "results_complete/MLPClassifiercomplete.csv"
rf =    "results_complete/RandomForestClassifiercomplete.csv"
xgb =   "results_complete/XGBClassifiercomplete.csv"

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

adab =  "results_complete/AdaBoostClassifierchikdengcomplete.csv"
gnb =   "results_complete/GaussianNBchikdengcomplete.csv"
gbc =   "results_complete/GradientBoostingClassifierchikdengcomplete.csv"
knn =   "results_complete/KNeighborsClassifierchikdengcomplete.csv"
mlp =   "results_complete/MLPClassifierchikdengcomplete.csv"
rf =    "results_complete/RandomForestClassifierchikdengcomplete.csv"
xgb =   "results_complete/XGBClassifierchikdengcomplete.csv"

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

adab =  "results_complete/AdaBoostClassifierdengundefcomplete.csv"
gnb =   "results_complete/GaussianNBdengundefcomplete.csv"
gbc =   "results_complete/GradientBoostingClassifierdengundefcomplete.csv"
knn =   "results_complete/KNeighborsClassifierdengundefcomplete.csv"
mlp =   "results_complete/MLPClassifierdengundefcomplete.csv"
rf =    "results_complete/RandomForestClassifierdengundefcomplete.csv"
xgb =   "results_complete/XGBClassifierdengundefcomplete.csv"

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

adab =  "results_complete/AdaBoostClassifierchikundefcomplete.csv"
gnb =   "results_complete/GaussianNBchikundefcomplete.csv"
gbc =   "results_complete/GradientBoostingClassifierchikundefcomplete.csv"
knn =   "results_complete/KNeighborsClassifierchikundefcomplete.csv"
mlp =   "results_complete/MLPClassifierchikundefcomplete.csv"
rf =    "results_complete/RandomForestClassifierchikundefcomplete.csv"
xgb =   "results_complete/XGBClassifierchikundefcomplete.csv"

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