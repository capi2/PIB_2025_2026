import pandas as pd


#TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results/AdaBoostClassifier_perclass.csv"
gnb =   "results/GaussianNB_perclass.csv"
gbc =   "results/GradientBoostingClassifier_perclass.csv"
knn =   "results/KNeighborsClassifier_perclass.csv"
mlp =   "results/MLPClassifier_perclass.csv"
rf =    "results/RandomForestClassifier_perclass.csv"
xgb =   "results/XGBClassifier_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp, gnb]

tab1 = ""
tab2 = ""

for file in result_files:
    df = pd.read_csv(file)

    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
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
        
print("Model,Chik Precision,Chik Recall,Chik f1-score, Dengue Precision, Dengue Recall, Dengue f1-score, Undef Precision, Undef Recall, Undef f1-score")
print(tab1)
print("desv pad")
print(tab2)


#DENGUE X CHIKUNGUNYA

print("DENGUE X CHIKUNGUNYA")

adab =  "results/AdaBoostClassifierdengchik_perclass.csv"
gnb =   "results/GaussianNBdengchik_perclass.csv"
gbc =   "results/GradientBoostingClassifierdengchik_perclass.csv"
knn =   "results/KNeighborsClassifierdengchik_perclass.csv"
mlp =   "results/MLPClassifierdengchik_perclass.csv"
rf =    "results/RandomForestClassifierdengchik_perclass.csv"
xgb =   "results/XGBClassifierdengchik_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp, gnb]

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



#DENGUE X NAO DEFINIDO

print("DENGUE X NAO DEFINIDO")

adab =  "results/AdaBoostClassifierdengundef_perclass.csv"
gnb =   "results/GaussianNBdengundef_perclass.csv"
gbc =   "results/GradientBoostingClassifierdengundef_perclass.csv"
knn =   "results/KNeighborsClassifierdengundef_perclass.csv"
mlp =   "results/MLPClassifierdengundef_perclass.csv"
rf =    "results/RandomForestClassifierdengundef_perclass.csv"
xgb =   "results/XGBClassifierdengundef_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp, gnb]

tab1 = ""
tab2 = ""

for file in result_files:
    df = pd.read_csv(file)

    columns = ["deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
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
        
print("Model, Dengue Precision, Dengue Recall, Dengue f1-score, Undefined Precision, Undefined Recall, Undefined f1-score")
print(tab1)
print("desv pad")
print(tab2)


#CHIKUNGUNYA X NAO DEFINIDO

print("CHIKUNGUNYA X NAO DEFINIDO")

adab =  "results/AdaBoostClassifierchikundef_perclass.csv"
gnb =   "results/GaussianNBchikundef_perclass.csv"
gbc =   "results/GradientBoostingClassifierchikundef_perclass.csv"
knn =   "results/KNeighborsClassifierchikundef_perclass.csv"
mlp =   "results/MLPClassifierchikundef_perclass.csv"
rf =    "results/RandomForestClassifierchikundef_perclass.csv"
xgb =   "results/XGBClassifierchikundef_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp, gnb]

tab1 = ""
tab2 = ""

for file in result_files:
    df = pd.read_csv(file)

    columns = ["chik_prec", "chik_rec", "chik_f1", "undef_prec", "undef_rec", "undef_f1"]
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
        
print("Model, Chikungunya Precision, Chikungunya Recall, Chikungunya f1-score, Undefined Precision, Undefined Recall, Undefined f1-score")
print(tab1)
print("desv pad")
print(tab2)