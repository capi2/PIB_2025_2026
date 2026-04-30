import pandas as pd

with open("results_clinicoporclasse.tex", "w") as f:

    ##########################################################
    # TODAS AS 3 CLASSES ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierclinical_perclass.csv"
    gnb =   "results_clinical/GaussianNBclinical_perclass.csv"
    gbc =   "results_clinical/GradientBoostingClassifierclinical_perclass.csv"
    knn =   "results_clinical/KNeighborsClassifierclinical_perclass.csv"
    mlp =   "results_clinical/MLPClassifierclinical_perclass.csv"
    rf =    "results_clinical/RandomForestClassifierclinical_perclass.csv"
    xgb =   "results_clinical/XGBClassifierclinical_perclass.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    j = 0
    k = 0
    chikdata = []
    dengdata = []
    undefdata = []
    for file in result_files:
        chikvalues = []
        dengvalues = []
        undefvalues = []

        chikstd_values = []
        dengstd_values = []
        undefstd_values = []

        df = pd.read_csv(file)
        df.columns = columns

        chikvalues.append(model_names[i])
        chikstd_values.append(model_names[i])
        i += 1
        for name in ["chik_prec", "chik_rec", "chik_f1"]:
            med = df[name].mean()
            std = df[name].std()
            chikvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        chikdata.append(chikvalues)

        dengvalues.append(model_names[j])
        dengstd_values.append(model_names[j])
        j += 1
        for name in ["deng_prec", "deng_rec", "deng_f1"]:
            med = df[name].mean()
            std = df[name].std()
            dengvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        dengdata.append(dengvalues)

        undefvalues.append(model_names[k])
        undefstd_values.append(model_names[k])
        k += 1
        for name in ["undef_prec", "undef_rec", "undef_f1"]:
            med = df[name].mean()
            std = df[name].std()
            undefvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        undefdata.append(undefvalues)

    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(chikdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Chikungunya", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Dengue", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DENGUE ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierchikdengclinical_perclass.csv"
    gnb =   "results_clinical/GaussianNBchikdengclinical_perclass.csv"
    gbc =   "results_clinical/GradientBoostingClassifierchikdengclinical_perclass.csv"
    knn =   "results_clinical/KNeighborsClassifierchikdengclinical_perclass.csv"
    mlp =   "results_clinical/MLPClassifierchikdengclinical_perclass.csv"
    rf =    "results_clinical/RandomForestClassifierchikdengclinical_perclass.csv"
    xgb =   "results_clinical/XGBClassifierchikdengclinical_perclass.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    j = 0
    chikdata = []
    dengdata = []
    for file in result_files:
        chikvalues = []
        dengvalues = []

        chikstd_values = []
        dengstd_values = []

        df = pd.read_csv(file)
        df.columns = columns

        chikvalues.append(model_names[i])
        chikstd_values.append(model_names[i])
        i += 1
        for name in ["chik_prec", "chik_rec", "chik_f1"]:
            med = df[name].mean()
            std = df[name].std()
            chikvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        chikdata.append(chikvalues)

        dengvalues.append(model_names[j])
        dengstd_values.append(model_names[j])
        j += 1
        for name in ["deng_prec", "deng_rec", "deng_f1"]:
            med = df[name].mean()
            std = df[name].std()
            dengvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        dengdata.append(dengvalues)

    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(chikdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Chikungunya - Chikungunya X Dengue", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Dengue - Chikungunya X Dengue", column_format='lccc'))

    f.write("\n\n")

    ##########################################################
    # DENGUE X DESCARTADO ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierdengundefclinical_perclass.csv"
    gnb =   "results_clinical/GaussianNBdengundefclinical_perclass.csv"
    gbc =   "results_clinical/GradientBoostingClassifierdengundefclinical_perclass.csv"
    knn =   "results_clinical/KNeighborsClassifierdengundefclinical_perclass.csv"
    mlp =   "results_clinical/MLPClassifierdengundefclinical_perclass.csv"
    rf =    "results_clinical/RandomForestClassifierdengundefclinical_perclass.csv"
    xgb =   "results_clinical/XGBClassifierdengundefclinical_perclass.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    j = 0
    k = 0
    dengdata = []
    undefdata = []
    for file in result_files:
        dengvalues = []
        undefvalues = []

        dengstd_values = []
        undefstd_values = []

        df = pd.read_csv(file)
        df.columns = columns

        dengvalues.append(model_names[j])
        dengstd_values.append(model_names[j])
        j += 1
        for name in ["deng_prec", "deng_rec", "deng_f1"]:
            med = df[name].mean()
            std = df[name].std()
            dengvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        dengdata.append(dengvalues)

        undefvalues.append(model_names[k])
        undefstd_values.append(model_names[k])
        k += 1
        for name in ["undef_prec", "undef_rec", "undef_f1"]:
            med = df[name].mean()
            std = df[name].std()
            undefvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        undefdata.append(undefvalues)

    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Dengue - Dengue X Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Descartado/Não Definido - Dengue X Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DESCARTADO ###################################
    ##########################################################

    adab =  "results_clinical/AdaBoostClassifierchikundefclinical_perclass.csv"
    gnb =   "results_clinical/GaussianNBchikundefclinical_perclass.csv"
    gbc =   "results_clinical/GradientBoostingClassifierchikundefclinical_perclass.csv"
    knn =   "results_clinical/KNeighborsClassifierchikundefclinical_perclass.csv"
    mlp =   "results_clinical/MLPClassifierchikundefclinical_perclass.csv"
    rf =    "results_clinical/RandomForestClassifierchikundefclinical_perclass.csv"
    xgb =   "results_clinical/XGBClassifierchikundefclinical_perclass.csv"

    result_files = [adab, rf, gbc, xgb, knn, mlp, gnb]
    columns = ["chik_prec", "chik_rec", "chik_f1", "undef_prec", "undef_rec", "undef_f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB", "KNN", "MLP", "NB"]

    i = 0
    j = 0
    k = 0
    chikdata = []
    undefdata = []
    for file in result_files:
        chikvalues = []
        dengvalues = []
        undefvalues = []

        chikstd_values = []
        undefstd_values = []

        df = pd.read_csv(file)
        df.columns = columns

        chikvalues.append(model_names[i])
        chikstd_values.append(model_names[i])
        i += 1
        for name in ["chik_prec", "chik_rec", "chik_f1"]:
            med = df[name].mean()
            std = df[name].std()
            chikvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        chikdata.append(chikvalues)

        undefvalues.append(model_names[k])
        undefstd_values.append(model_names[k])
        k += 1
        for name in ["undef_prec", "undef_rec", "undef_f1"]:
            med = df[name].mean()
            std = df[name].std()
            undefvalues.append(str(round(med, 4)) + " ± " + str(round(std, 4)))
        undefdata.append(undefvalues)

    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(chikdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Chikungunya - Chikungunya X Descartado/ Não Definido", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados Descartado/Não Definido - Chikungunya X Descartado/ Não Definido", column_format='lccc'))