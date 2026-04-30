import pandas as pd

with open("results_macro_clinico.tex", "w") as f:

    ##########################################################
    # TODAS AS 3 CLASSES ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierclinical.csv"
    gnb =   "results_clinical/GaussianNBclinical.csv"
    gbc =   "results_clinical/GradientBoostingClassifierclinical.csv"
    knn =   "results_clinical/KNeighborsClassifierclinical.csv"
    mlp =   "results_clinical/MLPClassifierclinical.csv"
    rf =    "results_clinical/RandomForestClassifierclinical.csv"
    xgb =   "results_clinical/XGBClassifierclinical.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["accuracy", "precision", "recall", "f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    data = []
    for file in result_files:
        values = []
        std_values = []
        df = pd.read_csv(file)
        df.columns = columns
        values.append(model_names[i])
        std_values.append(model_names[i])
        i += 1
        for name in columns:
            med = df[name].mean()
            std = df[name].std()
            values.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        data.append(values)

    res_columns = ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(data, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados macro clínico", column_format='lcccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DENGUE ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierchikdengclinical.csv"
    gnb =   "results_clinical/GaussianNBchikdengclinical.csv"
    gbc =   "results_clinical/GradientBoostingClassifierchikdengclinical.csv"
    knn =   "results_clinical/KNeighborsClassifierchikdengclinical.csv"
    mlp =   "results_clinical/MLPClassifierchikdengclinical.csv"
    rf =    "results_clinical/RandomForestClassifierchikdengclinical.csv"
    xgb =   "results_clinical/XGBClassifierchikdengclinical.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["accuracy", "precision", "recall", "f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    data = []
    for file in result_files:
        values = []
        std_values = []
        df = pd.read_csv(file)
        df.columns = columns
        values.append(model_names[i])
        std_values.append(model_names[i])
        i += 1
        for name in columns:
            med = df[name].mean()
            std = df[name].std()
            values.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        data.append(values)

    res_columns = ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(data, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados macro Chikungunya X Dengue", column_format='lcccc'))

    f.write("\n\n")

    ##########################################################
    # DENGUE X DESCARTADO ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierdengundefclinical.csv"
    gnb =   "results_clinical/GaussianNBdengundefclinical.csv"
    gbc =   "results_clinical/GradientBoostingClassifierdengundefclinical.csv"
    knn =   "results_clinical/KNeighborsClassifierdengundefclinical.csv"
    mlp =   "results_clinical/MLPClassifierdengundefclinical.csv"
    rf =    "results_clinical/RandomForestClassifierdengundefclinical.csv"
    xgb =   "results_clinical/XGBClassifierdengundefclinical.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["accuracy", "precision", "recall", "f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    data = []
    for file in result_files:
        values = []
        std_values = []
        df = pd.read_csv(file)
        df.columns = columns
        values.append(model_names[i])
        std_values.append(model_names[i])
        i += 1
        for name in columns:
            med = df[name].mean()
            std = df[name].std()
            values.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        data.append(values)

    res_columns = ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(data, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados macro Dengue X Descartado/Não Definido", column_format='lcccc'))

    f.write("\n\n")


    ##########################################################
    # CHIK X DESCARTADO ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierchikundefclinical.csv"
    gnb =   "results_clinical/GaussianNBchikundefclinical.csv"
    gbc =   "results_clinical/GradientBoostingClassifierchikundefclinical.csv"
    knn =   "results_clinical/KNeighborsClassifierchikundefclinical.csv"
    mlp =   "results_clinical/MLPClassifierchikundefclinical.csv"
    rf =    "results_clinical/RandomForestClassifierchikundefclinical.csv"
    xgb =   "results_clinical/XGBClassifierchikundefclinical.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["accuracy", "precision", "recall", "f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    data = []
    for file in result_files:
        values = []
        std_values = []
        df = pd.read_csv(file)
        df.columns = columns
        values.append(model_names[i])
        std_values.append(model_names[i])
        i += 1
        for name in columns:
            med = df[name].mean()
            std = df[name].std()
            values.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        data.append(values)

    res_columns = ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(data, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados macro Chikungunya X Descartado/Não Definido", column_format='lcccc'))