import pandas as pd

adab = "AdaBoostClassifier_perclass.csv"
gnb = "GaussianNB_perclass.csv"
gbc = "GradientBoostingClassifier_perclass.csv"
knn = "KNeighborsClassifier_perclass.csv"
mlp = "MLPClassifier_perclass.csv"
rf = "RandomForestClassifier_perclass.csv"
xgb = "XGBClassifier_perclass.csv"

result_files = [adab, gnb, gbc, knn, mlp, rf, xgb]

for file in result_files:
    df = pd.read_csv(file)

    columns = ["chik_prec", "chik_rec", "chik_f1", "deng_prec", "deng_rec", "deng_f1", "undef_prec", "undef_rec", "undef_f1"]
    df.columns = columns

    print(f"results for {file}")
    for name in columns:
        med = df[name].mean()
        std = df[name].std()
        print(f"\t{name} -> med {med} std {std}")
        
