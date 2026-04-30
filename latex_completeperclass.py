import pandas as pd

with open("results_completoporclasse.tex", "w") as f:

    ##########################################################
    # TODAS AS 3 CLASSES ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifiercomplete_perclass.csv"
    gnb =   "results_complete/GaussianNBcomplete_perclass.csv"
    gbc =   "results_complete/GradientBoostingClassifiercomplete_perclass.csv"
    knn =   "results_complete/KNeighborsClassifiercomplete_perclass.csv"
    mlp =   "results_complete/MLPClassifiercomplete_perclass.csv"
    rf =    "results_complete/RandomForestClassifiercomplete_perclass.csv"
    xgb =   "results_complete/XGBClassifiercomplete_perclass.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados completo - Chikungunya", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo - Dengue", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DENGUE ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierchikdengcomplete_perclass.csv"
    gnb =   "results_complete/GaussianNBchikdengcomplete_perclass.csv"
    gbc =   "results_complete/GradientBoostingClassifierchikdengcomplete_perclass.csv"
    knn =   "results_complete/KNeighborsClassifierchikdengcomplete_perclass.csv"
    mlp =   "results_complete/MLPClassifierchikdengcomplete_perclass.csv"
    rf =    "results_complete/RandomForestClassifierchikdengcomplete_perclass.csv"
    xgb =   "results_complete/XGBClassifierchikdengcomplete_perclass.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados completo Chikungunya - Chikungunya X Dengue", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo Dengue - Chikungunya X Dengue", column_format='lccc'))

    f.write("\n\n")

    ##########################################################
    # DENGUE X DESCARTADO ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierdengundefcomplete_perclass.csv"
    gnb =   "results_complete/GaussianNBdengundefcomplete_perclass.csv"
    gbc =   "results_complete/GradientBoostingClassifierdengundefcomplete_perclass.csv"
    knn =   "results_complete/KNeighborsClassifierdengundefcomplete_perclass.csv"
    mlp =   "results_complete/MLPClassifierdengundefcomplete_perclass.csv"
    rf =    "results_complete/RandomForestClassifierdengundefcomplete_perclass.csv"
    xgb =   "results_complete/XGBClassifierdengundefcomplete_perclass.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados completo Dengue - Dengue X Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo Descartado/Não Definido - Dengue X Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")


    ##########################################################
    # CHIKUNGUNYA X DESCARTADO ###################################
    ##########################################################

    adab =  "results_complete/AdaBoostClassifierchikundefcomplete_perclass.csv"
    gnb =   "results_complete/GaussianNBchikundefcomplete_perclass.csv"
    gbc =   "results_complete/GradientBoostingClassifierchikundefcomplete_perclass.csv"
    knn =   "results_complete/KNeighborsClassifierchikundefcomplete_perclass.csv"
    mlp =   "results_complete/MLPClassifierchikundefcomplete_perclass.csv"
    rf =    "results_complete/RandomForestClassifierchikundefcomplete_perclass.csv"
    xgb =   "results_complete/XGBClassifierchikundefcomplete_perclass.csv"

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
    f.write(res_df.to_latex(index=False, caption="Resultados completo Chikungunya - Chikungunya X Descartado/ Não Definido", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Modelo", "Precisão", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo Descartado/Não Definido - Chikungunya X Descartado/ Não Definido", column_format='lccc'))