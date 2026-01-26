import pandas as pd

adab = "AdaBoostClassifier.csv"
gnb = "GaussianNB.csv"
gbc = "GradientBoostingClassifier.csv"
knn = "KNeighborsClassifier.csv"
mlp = "MLPClassifier.csv"
rf = "RandomForestClassifier.csv"
xgb = "XGBClassifier.csv"

result_files = [adab, gnb, gbc, knn, mlp, rf, xgb]

for file in result_files:
    df = pd.read_csv(file)

    columns = ["accuracy", "precision", "recall", "f1"]
    df.columns = columns

    print(f"results for {file}")
    for name in columns:
        med = df[name].mean()
        std = df[name].std()
        print(f"\t{name} -> Med {med} std {std}")
        
