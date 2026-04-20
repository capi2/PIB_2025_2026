import pandas as pd


#TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results_clinical/AdaBoostClassifierclinical_perclass.csv"
gnb =   "results_clinical/GaussianNBclinical_perclass.csv"
gbc =   "results_clinical/GradientBoostingClassifierclinical_perclass.csv"
knn =   "results_clinical/KNeighborsClassifierclinical_perclass.csv"
mlp =   "results_clinical/MLPClassifierclinical_perclass.csv"
rf =    "results_clinical/RandomForestClassifierclinical_perclass.csv"
xgb =   "results_clinical/XGBClassifierclinical_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp]

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

adab =  "results_clinical/AdaBoostClassifierchikdengclinical_perclass.csv"
gnb =   "results_clinical/GaussianNBchikdengclinical_perclass.csv"
gbc =   "results_clinical/GradientBoostingClassifierchikdengclinical_perclass.csv"
knn =   "results_clinical/KNeighborsClassifierchikdengclinical_perclass.csv"
mlp =   "results_clinical/MLPClassifierchikdengclinical_perclass.csv"
rf =    "results_clinical/RandomForestClassifierchikdengclinical_perclass.csv"
xgb =   "results_clinical/XGBClassifierchikdengclinical_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp]

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

adab =  "results_clinical/AdaBoostClassifierdengundefclinical_perclass.csv"
gnb =   "results_clinical/GaussianNBdengundefclinical_perclass.csv"
gbc =   "results_clinical/GradientBoostingClassifierdengundefclinical_perclass.csv"
knn =   "results_clinical/KNeighborsClassifierdengundefclinical_perclass.csv"
mlp =   "results_clinical/MLPClassifierdengundefclinical_perclass.csv"
rf =    "results_clinical/RandomForestClassifierdengundefclinical_perclass.csv"
xgb =   "results_clinical/XGBClassifierdengundefclinical_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp]

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

adab =  "results_clinical/AdaBoostClassifierchikundefclinical_perclass.csv"
gnb =   "results_clinical/GaussianNBchikundefclinical_perclass.csv"
gbc =   "results_clinical/GradientBoostingClassifierchikundefclinical_perclass.csv"
knn =   "results_clinical/KNeighborsClassifierchikundefclinical_perclass.csv"
mlp =   "results_clinical/MLPClassifierchikundefclinical_perclass.csv"
rf =    "results_clinical/RandomForestClassifierchikundefclinical_perclass.csv"
xgb =   "results_clinical/XGBClassifierchikundefclinical_perclass.csv"

result_files = [adab, rf, gbc, gnb, xgb, knn, mlp]

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