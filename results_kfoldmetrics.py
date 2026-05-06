import pandas as pd

# RESULTADOS MACRO

with open("results_crossvalidation_macro_complete.tex", "w") as f:

    adab =  "results_macro_crossvalidation/AdaBoostClassifiercrossvalidationmacrocomplete"
    rf =    "results_macro_crossvalidation/RandomForestClassifiercrossvalidationmacrocomplete"
    gbc =   "results_macro_crossvalidation/GradientBoostingClassifiercrossvalidationmacrocomplete"
    xgb =   "results_macro_crossvalidation/XGBClassifiercrossvalidationmacrocomplete"

    result_files = [adab, rf, gbc, xgb]
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

    res_columns = ["Model", "Accuracy", "Precision", "Recall", "F1-score"]
    res_df = pd.DataFrame(data, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados macro completo", column_format='lcccc'))

    f.write("\n\n")


# RESULTADOS POR CLASS

with open("results_crossvalidation_perclass_complete.tex", "w") as f:

    adab =  "results_crossvalidation_perclass/AdaBoostClassifiercrossvalidation_perclasscomplete"
    gbc =   "results_crossvalidation_perclass/GradientBoostingClassifiercrossvalidation_perclasscomplete"
    rf =    "results_crossvalidation_perclass/RandomForestClassifiercrossvalidation_perclasscomplete"
    xgb =   "results_crossvalidation_perclass/XGBClassifiercrossvalidation_perclasscomplete"

    result_files = [adab, rf, gbc, xgb]
    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
    model_names = ["Adaboost", "RF", "GBM", "XGB"]

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

    res_columns = ["Model", "Precision", "Recall", "F1-score"]
    res_df = pd.DataFrame(chikdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo - Chikungunya", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Model", "Precision", "Recall", "F1-score"]
    res_df = pd.DataFrame(dengdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo - Dengue", column_format='lccc'))

    f.write("\n\n")


    res_columns = ["Model", "Precision", "Recall", "F1-score"]
    res_df = pd.DataFrame(undefdata, columns=res_columns)
    f.write(res_df.to_latex(index=False, caption="Resultados completo Descartado/Não Definido", column_format='lccc'))

    f.write("\n\n")