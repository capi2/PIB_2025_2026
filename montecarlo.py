import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

dengue = "dengue.csv"
undef = "undef.csv"
chik = "chik.csv"

def random_dataset(denguefile, undeffile, chikfile, s):
    df = []
    skip = sorted(random.sample(range(1, 4307513+1), 4307513-s))
    df.append(pd.read_csv(denguefile, skiprows=skip))

    #skip = sorted(random.sample(range(1, 2100029+1), 2100029-s))
    #df.append(pd.read_csv(undeffile, skiprows=skip))

    df.append(pd.read_csv(chikfile))

    return pd.concat(df)

def process_dataset(df):
    cols = ['ID_AGRAVO', "DT_NOTIFIC", "SEM_NOT", "NU_ANO", "SG_UF_NOT", "ID_MUNICIP", "ID_REGIONA", "ID_UNIDADE", "DT_SIN_PRI", "SEM_PRI", "SG_UF",
            "ID_MN_RESI", "ID_RG_RESI", "VOMITO", "LACO", "ID_PAIS", "DT_INVEST", "TPAUTOCTO", "COUFINF", "COPAISINF", "COMUNINF", "DT_ENCERRA", "DT_NASC",
            "RESUL_SORO", "RESUL_NS1", "RESUL_VI_N", "RESUL_PCR_", "HISTOPA_N", "IMUNOH_N", "LEUCOPENIA", 'CLASSI_FIN'] # removido vomito e laco

    x = df.drop(cols, axis=1)
    y = df['CLASSI_FIN']

    # classes ['Chikungunya' 'Dengue' 'Discarded/Inconclusive']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    print(le.classes_)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #print(x,y)
    
    return x_train, x_test, y_train, y_test

def run_classifier(classifier):
    results_file = open(f"{classifier.__class__.__name__}dengchik_perclass.csv", "a")
    #results_file.write("num, accuracy, precision, recall, f1")

    sample = 325_000 # undersample para classe chikungunya
    
    df = random_dataset(dengue, undef, chik, sample)
    print(df.shape)
    
    x_train, x_test, y_train, y_test = process_dataset(df)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results_file.write(f"{acc}, {prec}, {rec}, {f1}\n")
    """
    #labels = ['Chikungunya', 'Dengue']
    labels = [0, 1]
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)

    chik_prec = prec[0]
    chik_rec = rec[0]
    chik_f1 = f1[0]

    deng_prec = prec[1]
    deng_rec = rec[1]
    deng_f1 = f1[1]
    """
    undef_prec = prec[2]
    undef_rec = rec[2]
    undef_f1 = f1[2]
    """
    # chik -> dengue -> undef
    # precisao -> recall -> f1
    results_file.write(f"{chik_prec}, {chik_rec}, {chik_f1}, {deng_prec}, {deng_rec}, {deng_f1}\n")
    

def main():
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.1)
    
    run_classifier(clf)

main()