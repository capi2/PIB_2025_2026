import pandas as pd

with open("results_macro_complete.tex", "w") as f:

    ##########################################################
    # TODAS AS 3 CLASSES ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifiercomplete.csv"
    gnb =   "results_complete/GaussianNBcomplete.csv"
    gbc =   "results_complete/GradientBoostingClassifiercomplete.csv"
    knn =   "results_complete/KNeighborsClassifiercomplete.csv"
    mlp =   "results_complete/MLPClassifiercomplete.csv"
    rf =    "results_complete/RandomForestClassifiercomplete.csv"
    xgb =   "results_complete/XGBClassifiercomplete.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados macro completo", column_format='lcccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DENGUE ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierchikdengcomplete.csv"
    gnb =   "results_complete/GaussianNBchikdengcomplete.csv"
    gbc =   "results_complete/GradientBoostingClassifierchikdengcomplete.csv"
    knn =   "results_complete/KNeighborsClassifierchikdengcomplete.csv"
    mlp =   "results_complete/MLPClassifierchikdengcomplete.csv"
    rf =    "results_complete/RandomForestClassifierchikdengcomplete.csv"
    xgb =   "results_complete/XGBClassifierchikdengcomplete.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados macro completo Chikungunya X Dengue", column_format='lcccc'))

    f.write("\n\n")

    ##########################################################
    # DENGUE X DESCARTADO ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierdengundefcomplete.csv"
    gnb =   "results_complete/GaussianNBdengundefcomplete.csv"
    gbc =   "results_complete/GradientBoostingClassifierdengundefcomplete.csv"
    knn =   "results_complete/KNeighborsClassifierdengundefcomplete.csv"
    mlp =   "results_complete/MLPClassifierdengundefcomplete.csv"
    rf =    "results_complete/RandomForestClassifierdengundefcomplete.csv"
    xgb =   "results_complete/XGBClassifierdengundefcomplete.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados macro completo Dengue X Descartado/Não Definido", column_format='lcccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DESCARTADO ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierchikundefcomplete.csv"
    gnb =   "results_complete/GaussianNBchikundefcomplete.csv"
    gbc =   "results_complete/GradientBoostingClassifierchikundefcomplete.csv"
    knn =   "results_complete/KNeighborsClassifierchikundefcomplete.csv"
    mlp =   "results_complete/MLPClassifierchikundefcomplete.csv"
    rf =    "results_complete/RandomForestClassifierchikundefcomplete.csv"
    xgb =   "results_complete/XGBClassifierchikundefcomplete.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados macro completo Chikungunya X Descartado/Não Definido", column_format='lcccc'))