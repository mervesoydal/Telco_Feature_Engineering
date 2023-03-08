#########################################
#Telco Churn Feature Engineering
#########################################

# Problem Tanımı:

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi istenmektedir.  Modeli geliştirmeden öncegerekliolan veri
# analizi veözellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

#Veri Seti Hikayesi:

#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye
# ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi
# hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını,
# kaldığını veya hizmete kaydolduğunu gösterir.


# TASK 1: Keşifçi Veri Analizi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data

df = load()
df.head()
df.shape
df.info()


"""
CustomerId :Müşteri İd’si 
Gender :Cinsiyet 
SeniorCitizen :Müşterinin yaşlı olup olmadığı(1, 0) 
Partner :Müşterinin bir ortağı olup olmadığı(Evet, Hayır) 
Dependents :Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır)
tenure: Müşterinin şirkette kaldığı ay sayısı 
PhoneService: Müşterinin telefon hizmeti olup olmadığı(Evet, Hayır) 
MultipleLines :Müşterinin birden fazla hattı olup olmadığı(Evet, Hayır, Telefonhizmeti yok) 
InternetService :Müşterinin internet servis sağlayıcısı(DSL, Fiber optik, Hayır) 
OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı(Evet, Hayır, İnternet hizmeti yok) 
OnlineBackup :Müşterinin online yedeğinin olup olmadığı(Evet, Hayır, İnternet hizmeti yok) 
DeviceProtection :Müşterinin cihaz korumasına sahip olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
TechSupport :Müşterinin teknik destek alıp almadığı(Evet, Hayır, İnternet hizmeti yok) 
StreamingTV :MüşterininTV yayını olup olmadığı(Evet, Hayır, İnternet hizmeti yok) 
StreamingMovies :Müşterinin film akışı olup olmadığı(Evet, Hayır, İnternet hizmeti yok) 
Contract : Müşterinin sözleşme süresi(Aydan aya, Bir yıl, İkiyıl) 
PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı(Evet, Hayır) 
PaymentMethod: Müşterinin ödeme yöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik), Kredikartı(otomatik)) 
MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar 
TotalCharges: Müşteriden tahsil edilentoplamtutar 
Churn :Müşterininkullanıpkullanmadığı(Evet veya Hayır)

"""
#STEP 1.1:
#Normalde float olması gerekirken object tipinde olduğu için TotalCharges değişkenine bir dönüşüm işlemi
#uygulandı.

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0 )
df.info()

# Dönüşüm sonrası:
# 19  TotalCharges      7032 non-null   float64
# 20  Churn             7043 non-null   int64

# STEP 1.2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Observations: 7043
#Variables: 21
#cat_cols: 17
#num_cols: 3
#cat_but_car: 1
#num_but_cat: 2

# STEP 1.3: Numerik ve kategorik değişkenlerin analizini yapalım.

# Kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=False)

#['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
# 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen', 'Churn']

#Numerik değişkenlerin analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

#STEP 1.4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
#hedef değişkene göre numerik değişkenlerin ortalaması)

# Numerik değişkenler için:
def target_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))

for col in num_cols:
 target_with_num(df,"Churn",col)

# num_cols
# ['tenure', 'MonthlyCharges', 'TotalCharges']

# STEP 1.5: KORELASYON ANALİZİ
#korelasyon matrisi

df.corr()

#Matrisi Görselleştirme
sns.heatmap(df.corr(),
    annot=True, fmt=".2g")
plt.show()

# STEP 1.6: #Eksik Değer Analizi
# eksik gozlem var mı yok mu sorgusu
df.isnull().sum()

# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        11
# Churn               0

df.isnull().values.any()
#True

# Sonuçlardan da anlaşılacağı üzere veri setinde eksik değer vardır.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name = True)

#    n_miss               ratio
#   TotalCharges      11  0.160

# Eksik değer sadece TotalCharges da olduğu için median veya mod değerleri ile doldurabiliriz.

df['TotalCharges'].fillna(df['TotalCharges'].median().round(1), inplace = True)

df.isnull().sum()

"""
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
"""

# Aykırı değer analizi yapmadan bu haliyle modelimizi kurup eğitelim ve aykırı değer analizi yaptıktan sonraki
#durum ile karşılaştıralım.

#Encoding:

# ENCODING

df2 = df.copy()


# One hot encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

new_cat_cols = [col for col in cat_cols not in ["Churn"]]
df2.head()

#new_cat_cols
#['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
# 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'SeniorCitizen']
df2 = one_hot_encoder(df2, new_cat_cols, drop_first= True)

#Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df2.columns if df2[col].dtype not in [int, float]
               and df2[col].nunique() == 2]

for col in binary_cols:
    df2 = label_encoder(df2, col)
df2.head()

"""
Encoder uyguladıktan sonra:
  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  SeniorCitizen_1  
0                                    0                    0                  0                  0                                      0                               1                           0                0  
1                                    0                    0                  1                  0                                      0                               0                           1                0  
2                                    0                    0                  0                  0                                      0                               0                           1                0  
3                                    0                    0                  1                  0                                      0                               0                           0                0  
4                                    0                    0                  0                  0                                      0                               1                           0                0  
"""

#Modelleme

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)})")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)})")
print(f"F1: {round(f1_score(y_pred, y_test), 4)})")

"""
# Aykırı değer analizi yapmadan sonuç:

Accuracy: 0.7851
Recall: 0.6395)
Precision: 0.4791)
F1: 0.5478)
"""

#STEP 1.7: AYKIRI DEĞER ANALİZİ
# 1. Eşik değer belirleme
# 2. Aykırılara eriştik.
# 3. Aykırı değer var mı yok diye bakıldı.


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



# Outlier değerleri tespit ettikten sonra threshould değerleri ile değiştiriyoruz.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df2, col))

# tenure False
# MonthlyCharges False
# TotalCharges False
#Herhangi bir aykırı değer olmadığı için replace_with_threshould fonksiyonunu kullanmaya gerek kalmadı.

for col in num_cols:
    replace_with_thresholds(df2, col)

# Yeni değişkenler oluşturalım.

df.loc[(df['tenure'] > 0) & (df['tenure'] < 12), 'New_Year_Tenure'] = "0-1 years"
df.loc[(df['tenure'] > 12) & (df['tenure'] < 24), 'New_Year_Tenure'] = "1-2 years"
df.loc[(df['tenure'] > 24) & (df['tenure'] < 36), 'New_Year_Tenure'] = "2-3 years"
df.loc[(df['tenure'] > 36) & (df['tenure'] < 48), 'New_Year_Tenure'] = "3-4 years"
df.loc[(df['tenure'] > 48) & (df['tenure'] < 96), 'New_Year_Tenure'] = "4-5 years"

df["New_Senior_Tenure"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 1) and (x["tenure"] == 3) else 0, axis=1)
df["New_Senior_StreamingMovies"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 0) and (x["StreamingMovies"] == 1) else 0, axis=1)
df["New_yesProt"] = df.apply(lambda x: 1 if x["TechSupport"] == "Yes" or x["OnlineBackup"] == "Yes" or x["OnlineSecurity"] == "Yes" else 0 , axis = 1)
df["New_noAutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer(automatic)", "Credit card(automatic)"] else 0)
df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 7043
Variables: 25
cat_cols: 21
num_cols: 3
cat_but_car: 1
num_but_cat: 5
"""
#Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
#binary_cols
#['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn',
# 'New_Senior_Tenure', 'New_yesProt']

for col in binary_cols:
    df = label_encoder(df, col)

# One hot encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

new_cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]

#['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
# 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'New_Year_Tenure',
# 'New_Senior_StreamingMovies']

df.head()
df.info()

#new_cat_cols
df.head()
df= one_hot_encoder(df, new_cat_cols, drop_first= True)

#Modelleme

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(verbose=False, random_state=17).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
accuracy_score(y_pred, y_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)})")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)})")
print(f"F1: {round(f1_score(y_pred, y_test), 4)})")

"""

Accuracy: 0.7866
Recall: 0.6376
Precision: 0.4965
F1: 0.5583

"""