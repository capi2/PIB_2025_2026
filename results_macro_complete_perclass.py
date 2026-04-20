import pandas as pd

print("RESULTADOS COM QUASE TODAS AS FEATURES PARA CADA CLASSE")

#TODAS AS 3 CLASSES

print("TODAS AS 3 CLASSES")

adab =  "results_complete/AdaBoostClassifiercomplete_perclass.csv"
gnb =   "results_complete/GaussianNBcomplete_perclass.csv"
gbc =   "results_complete/GradientBoostingClassifiercomplete_perclass.csv"
knn =   "results_complete/KNeighborsClassifiercomplete_perclass.csv"
mlp =   "results_complete/MLPClassifiercomplete_perclass.csv"
rf =    "results_complete/RandomForestClassifiercomplete_perclass.csv"
xgb =   "results_complete/XGBClassifiercomplete_perclass.csv"

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

adab =  "results_complete/AdaBoostClassifierchikdengcomplete_perclass.csv"
gnb =   "results_complete/GaussianNBchikdengcomplete_perclass.csv"
gbc =   "results_complete/GradientBoostingClassifierchikdengcomplete_perclass.csv"
knn =   "results_complete/KNeighborsClassifierchikdengcomplete_perclass.csv"
mlp =   "results_complete/MLPClassifierchikdengcomplete_perclass.csv"
rf =    "results_complete/RandomForestClassifierchikdengcomplete_perclass.csv"
xgb =   "results_complete/XGBClassifierchikdengcomplete_perclass.csv"

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

adab =  "results_complete/AdaBoostClassifierdengundefcomplete_perclass.csv"
gnb =   "results_complete/GaussianNBdengundefcomplete_perclass.csv"
gbc =   "results_complete/GradientBoostingClassifierdengundefcomplete_perclass.csv"
knn =   "results_complete/KNeighborsClassifierdengundefcomplete_perclass.csv"
mlp =   "results_complete/MLPClassifierdengundefcomplete_perclass.csv"
rf =    "results_complete/RandomForestClassifierdengundefcomplete_perclass.csv"
xgb =   "results_complete/XGBClassifierdengundefcomplete_perclass.csv"

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

adab =  "results_complete/AdaBoostClassifierchikundefcomplete_perclass.csv"
gnb =   "results_complete/GaussianNBchikundefcomplete_perclass.csv"
gbc =   "results_complete/GradientBoostingClassifierchikundefcomplete_perclass.csv"
knn =   "results_complete/KNeighborsClassifierchikundefcomplete_perclass.csv"
mlp =   "results_complete/MLPClassifierchikundefcomplete_perclass.csv"
rf =    "results_complete/RandomForestClassifierchikundefcomplete_perclass.csv"
xgb =   "results_complete/XGBClassifierchikundefcomplete_perclass.csv"

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