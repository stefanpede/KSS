import numpy as np
import pandas as pd
import random

def dropCorrelatedColumns(sets, covBound):
    from scipy import stats

    ## suche hohe Werte in Kovarianz-Matrix
    if covBound < 0:
        text = "korrelieren negativ"
        kovBool = pd.DataFrame(np.cov((stats.zscore(sets[0]).T)) < covBound)
    else:
        text = "korrelieren positiv"
        kovBool = pd.DataFrame(np.cov((stats.zscore(sets[0]).T)) > covBound)        
    ## suche diejenigen, die nicht auf Diagonale liegen
    korr = []
    for a,b in zip(np.where(kovBool)[0], np.where(kovBool)[1]):
        if (a != b):
            korr.append([a,b])
    print(korr)
    ## sortiere diese und finde einzigartige
    korr = [sorted(i) for i in korr]
    korrWD = []
    for i in korr:
        if i not in korrWD:
            korrWD.append(i)
            print(sets[0].columns[i[0]]," und ",sets[1].columns[i[1]],text)
    ## erhalte Indizes der korrelierenden Spalten
    drop = []
    for i in korrWD:
        drop.append(i[1])

    ## lösche diese aus den vorliegenden Sets           
    retTrainingSet = sets[0].drop(sets[0].columns[drop], axis=1)
    retTestSet = sets[1].drop(sets[1].columns[drop], axis=1)
    return(retTrainingSet, retTestSet)

def preprocess(df, spaltenSchranke=15, zeilenSchranke=5):
    relNaNsCol = np.sum(np.isnan(df))/df.shape[0]*100
    # schmeiße zunächst alle Spalten heraus, die mehr als bestimmte Prozent an NaNs haben
    
    keep = [i for i in np.arange(len(relNaNsCol)) if relNaNsCol[i] <= spaltenSchranke]
    dfVV = df[df.columns[keep]] # extrahiere Spalten

    # gleiches auf Zeilen anwenden
    relNaNsRow = dfVV.isnull().sum(axis=1)/dfVV.shape[1]*100
    keep = [i for i in np.arange(len(relNaNsRow)) if relNaNsRow[i] <= zeilenSchranke]
    dfVV2 = dfVV.iloc[keep] #extraheire Zeilen

    #übrige NaNs mit Mittelwert aus Spalten auffüllen
    dfVV2 = dfVV2.fillna(dfVV2.mean())
    
    return dfVV2

def get_product(df, index):
    groupby_list = ["Header_Leitguete","Header_Soll_AD","Header_Soll_WD"]
    products = list(df.groupby(groupby_list)[groupby_list].groups.keys())
    
    return groupby_list, products[index]

def get_data(df, label_encoder, test_frac=0.2):
    train_set = {'data': [], 'label': []}
    test_set = {'data': [], 'label': []}


    df_walzlos = df.groupby(["Header_Walzlos"])["Header_Pseudonummer"].agg(["count"])
    df_without_header = df[df.columns[6:]]

    for walzlos in np.unique(df["Header_Walzlos"]):
        num_walzlos = np.int(df_walzlos.ix[walzlos])

        a = df_without_header.ix[df["Header_Walzlos"] == walzlos]

        test_num = int(num_walzlos * test_frac)
        train_num = num_walzlos - test_num

        rows = random.sample(list(a.index), test_num)
        test_split = a.ix[rows]
        train_split = a.drop(rows)

        label = label_encoder.transform(walzlos)

        train_set['data'].extend(np.asarray(train_split))
        train_set['label'] += train_num*[label]

        test_set['data'].extend(np.asarray(test_split))
        test_set['label'] += test_num*[label]
    
    train_set['data'] = np.asarray(train_set['data'])
    test_set['data'] = np.asarray(test_set['data'])
        
    return train_set, test_set

def zscore(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0, keepdims=True)
    
    if std is None:
        std = np.std(data, axis=0, keepdims=True)
        
    normed_data = (data-mean)/std
    
    return normed_data, mean, std