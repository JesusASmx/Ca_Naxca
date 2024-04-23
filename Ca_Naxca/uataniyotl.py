import pandas as pd

'''
class csv_crossvals:
    def __init__(self, train_path, target_column, folds):
        self.train_path = train_path
        self.target_column = target_column
        self.folds = folds

    def get(self):
        df = pd.read_csv(self.train_path)
        df[self.target_column].value_counts()

        CÓMO APEÑUSCAR VALORES EN TÉRMINOS DE REGRESSIÓN??
        La idea es que, si graficamos ID vs Label, los resultados se van a parecer bastante.

        Para cada valor:
            -¿Hay #folds de ellos? Si sí, ponemos un sample en cada fold. (O varios. E.G. 2 si hay 10, con fold==5))
            Si no, simulamos iguales hasta tener una cantidad divisible por folds

def cv(csv, label_col, labels_ids, folds): #category
    final_csvs = []
    lista_csv = []
    for x in labels_ids:
        lab_csv = csv[csv[label_col]==x].copy().sample(frac=1).reset_index(drop=True)
        try:
            lista_csv.append(np.array_split(lab_csv, folds))
        except:
            print("CV detected as 0, we return training concat with validation")
            final_csvs.append([train_df])
            break

    # Ensemble the complete csv's:
    final_folds = []
    for i in range(folds):
        a = pd.concat([x[i].copy() for x in lista_csv]).sample(frac=1).reset_index(drop=True)
        final_folds.append(a)

    for i in range(folds):
        try:
            final_csvs.append(
                [
                    pd.DataFrame(pd.concat([final_folds[j].copy() for j in range(folds) if j != i])).sample(frac=1).reset_index(drop=True),
                    pd.DataFrame(final_folds[i].copy()).sample(frac=1).reset_index(drop=True)
                ]
            ) #[train_df, val_df]
        except:
            if folds == 1:
                print("CV detected as 1, we return training and validation sets instead.")
                entrena = pd.read_csv("..//Dataset//EDI_train.csv")
                validac = pd.read_csv("..//Dataset//EDI_val.csv")
                final_csvs.append([entrena, validac])
            break

    return final_csvs
'''

from torch.utils.data import Dataset

class csv_dataset(Dataset):
    def __init__(self, csv_dataframe, text_column, target_column):
        self.texts = []
        self.labels = []

        for x in csv_dataframe.index:
            self.texts.append(csv_dataframe[text_column][x])
            self.labels.append(csv_dataframe[target_column][x])

        self.n_examples = len(self.texts)
        return

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        if self.target:
            return {'text':self.texts[item], 'label':self.labels[item]}
        else:
            return {'text':self.texts[item]}



class collator_regression(object): #IMPORTANT: Labels must be already in a numeric type.
    def __init__(self, use_tokenizer, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        return

    def __call__(self, sequences):
        labels = [sequence['label'] for sequence in sequences]
        texts = [sequence['text'] for sequence in sequences]

        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})
        return inputs


#from sklearn.metrics import median_absolute_error
#from sklearn.metrics import mean_absolute_percentage_error

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

import matplotlib.pyplot as plt

class regression_report:
    def __init__(self, y_test, y_pred, IDs): #IDs es lista t.q. IDs[x] es el ID del sample con label y_test[x] y predicción y_pred[x]
        if len(y_test) != len(y_pred):
            raise Exception("Test and Prediction samples ammount does not match!")
        if len(y_test) != len(IDs):
            raise Exception("ID's ammount does not match with test and prediction!")
        
        self.IDs = IDs
        self.length = len(y_test)

        self.y_test = y_test
        self.y_pred = y_pred
        #self.diferencias = [abs(y_test[x] - y_pred[x]) for x in range(self.length)))]

    def display(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)

        r_squared = r2_score(self.y_test, self.y_pred)
        spearman = stats.spearmanr(self.y_test, self.y_pred)
        pearson = stats.pearsonr(self.y_test, self.y_pred)
        kendalltau = stats.kendalltau(self.y_test, self.y_pred)

        #mdae = median_absolute_error(y_test, y_pred)
        #mdape = ((pd.Series(y_test) - pd.Series(y_pred)) / pd.Series(y_test)).abs().median()    
        #mape = mean_absolute_percentage_error(y_test, y_pred)
        
        df = pd.DataFrame.from_dict({
                                #"Median Absolute Error": [mdae],
                                #"Mean Absolute Percentage Error": [mape],
                                #"MDAPE": [mdape],
                                 "Mean Absolute Error": [mae],
                                 "Mean Squared Error": [mse],
                                 "R2 score": [r_squared],
                                 "Spearman": [spearman],
                                 "Pearson": [pearson],
                                 "Kendall-Tau": [kendalltau]})
        return df


    def plot(self, path):
        cuantos = range(len(self.IDs))

        plt.figure(figsize=(10, 6))

        plt.scatter(cuantos, self.y_test, color='blue', label='Real Values', marker='o')
        plt.scatter(cuantos, self.y_pred, color='red', label='Pred. Values', marker='x')

        plt.xticks(cuantos, self.IDs, rotation=45)
        plt.xlabel('IDs')
        plt.ylabel('Labels')
        plt.title('Predicted vs Real values')
        plt.legend()

        plt.tight_layout()
        plt.savefig(path)
        
        plt.show()
