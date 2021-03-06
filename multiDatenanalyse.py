import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from collections import OrderedDict
import operator

from scipy.stats import zscore

# high level functions
def load_data(mmPfad='../data/Messmatrix.csv', cor_th=0.8, verbose=True):
pd.options.mode.chained_assignment = None

# high level functions
def load_data(mmPfad, cor_th=0.8):
    
    # Rohe Daten einladen
    df = pd.read_csv(mmPfad)
    if verbose:
        print("Rohe Daten: \n")
        print("Anzahl der Kennwerte: "+str(df.shape[1]))
        print("Anzahl der vermessenen Rohre: "+str(df.shape[0]))
        print("Anzahl der gefahrenen Produkte: "+str(df.groupby(["Header_Leitguete","Header_Soll_AD","Header_Soll_WD"])["Header_Pseudonummer"].agg(["count"]).shape[0]))
        print("Anzahl der Walzlose: "+str(len(pd.unique(df["Header_Walzlos"]))))
    
    # Vorberarbeitung, beseitigen von NaNs
    dfVV2 = preprocess(df)
    
    if verbose:
        print("\nDaten nach Vorverarbeitung: \n")
        print("Anzahl der Kennwerte: "+str(dfVV2.shape[1]))
        print("Anzahl der vermessenen Rohre: "+str(dfVV2.shape[0]))
        print("Anzahl der gefahrenen Produkte: "+str(dfVV2.groupby(["Header_Leitguete","Header_Soll_AD","Header_Soll_WD"])["Header_Pseudonummer"].agg(["count"]).shape[0]))
        print("Anzahl der Walzlose: "+str(len(pd.unique(dfVV2["Header_Walzlos"]))))

        # Korrelierte Merkmale entfernen
        print("\nKorrelierte Merkmale entfernen\n")
    dfNoCor, _ = dropCorrelatedColumns((dfVV2[dfVV2.columns[6:]], dfVV2[dfVV2.columns[6:]]), cor_th)
    dfVV2 = pd.concat((dfVV2[dfVV2.columns[:6]], dfNoCor), axis=1)
    
    return dfVV2

def extract_product(df, product_id=0, min_num_walzlos=100):
    # get product
    groupby_list, product = get_product(df, product_id)
    df_prod = df.query(" & ".join(["({} == {})".format(name, param) for name, param in zip(groupby_list, product)]))
    
    df_walzlos = df_prod.groupby(["Header_Walzlos"])["Header_Pseudonummer"].agg(["count"])
    
    # drop 
    walzlose_to_drop = df_walzlos[(df_walzlos['count'] < min_num_walzlos)].index.tolist()
    for walzlos in walzlose_to_drop:
        df_prod.drop(df_prod[df_prod["Header_Walzlos"] == walzlos].index, inplace=True)
        
    return df_prod

def plot_lda(X_lda, y, title, ax=None):
    if ax:
        ax = ax
    else:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111, projection='3d')
        
    for color, label in enumerate(np.unique(y)):
        min_val = 0
        max_val = len(np.unique(y))

        my_cmap = plt.cm.get_cmap('rainbow') # or any other one
        norm = matplotlib.colors.Normalize(min_val, max_val) # the color maps work for [0, 1]

        color_i = my_cmap(norm(color)) # returns an rgba value

        ax.scatter(X_lda[:,0][y==label], X_lda[:,1][y==label], X_lda[:,2][y==label], marker='*', color=color_i,
                    label=label, alpha=1)
        
    ax.set_xlabel('Komponente 1')
    ax.set_ylabel('Komponente 2')
    ax.set_zlabel('Komponente 3')
    ax.set_title(title)
    
    #ax.legend()
    ax.grid()                   
    

# basic functions
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

def get_lda_data(df, test_frac=0.2):
    label_encoder = LabelEncoder().fit(df["Header_Walzlos"])
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


def biplot(score,coeff,pcax,pcay,labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5) 
        if labels is None:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    
def mv_features(df_all_prod):
    features = list(df_all_prod[0].columns[6:])
    feature_dict = OrderedDict((feature,0) for feature in features)

    for product_id in range(len(df_all_prod)):
        df_prod = df_all_prod[product_id]

        if len(pd.unique(df_prod["Header_Walzlos"])) < 15:
            continue
        # train and test set
        test_frac = 0.05
        train_set, test_set = get_lda_data(df_prod, test_frac=test_frac)

        # zscore data
        train_set['data'] = zscore(train_set['data'])
        test_set['data'] = zscore(test_set['data'])

        # extract data and label
        X_train, y_train = train_set['data'], train_set['label']

        # LDA object
        sklearn_LDA = LDA(solver='eigen')

        # fit with train data
        sklearn_LDA = sklearn_LDA.fit(X_train, y_train)

        ew_ratio = sklearn_LDA.explained_variance_ratio_
        eigvecs = sklearn_LDA.scalings_

        for index_vec, eig_vec in enumerate(eigvecs.T):
            for index_feat, feature in enumerate(features):
                feature_dict[feature] += ew_ratio[index_vec] * np.abs(eig_vec[index_feat])

        sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_feature_dict