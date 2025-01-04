# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:12:42 2024

@author: Alparslan
"""

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.linear_model import ElasticNetCV  
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, classification_report  
import matplotlib.pyplot as plt  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import OneHotEncoder  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score  
from statsmodels.stats.outliers_influence import variance_inflation_factor  
from statsmodels.tools.tools import add_constant  
from statsmodels.api import Logit  

data = pd.read_excel("C:\\Users\\Alparslan\\Desktop\\verimadenvize\\vmv.xlsx")  
datatest = pd.read_excel("C:\\Users\\Alparslan\\Desktop\\verimadenvize\\vmvtest.xlsx")

###VERİ ÖN İŞLEME

#tanisuresiyıl değişkenini tanısuresiay değişkeni cinsinden yazıp, tek bir değişken değişken olarak tanımlıyoruz. 
data['tanisure_ay'] = (data['tanisuresiyil'] * 12) + data['tanisuresiay'] 
#acilservistoplamyatisgun değişkenini acilservistoplamyatissaat cinsinden yazıp tek bir değişken olarak tanımlıyoruz.  
data['acilservisetoplamyatissuresi_saat']= (data['acilservisetoplamyatıssuresigun']* 24)+ data['acilservisetoplamyatissuresisaat']
data['boy'] = data['boy'] / 100  # cm'den metreye çevirme  
# VKI hesaplama  
data['vki'] = data['vucutagirligi'] / (data['boy'] ** 2)  

#verimizde yer alan ancak gereksiz olan değişkenleri baştan çıkaralım.
data = data.drop(columns=['hastaNo','basvurutarihi','tanisuresiyil','tanisuresiay','acilservisetoplamyatıssuresigun','acilservisetoplamyatissuresisaat','FEV1','PEF','yogunbakimatoplamyatissuresisaat','servisetoplamyatissuresisaat','boy', 'vucutagirligi'])  
 
numerik_degiskenler = ['yas','sigarayibirakannekadargunicmis','sigarabirakangundekacadeticmis',   
'nezamanbirakmisgun','sigarayadevamedengundekacadeticiyo','acilserviseyatissayisi','yogunbakimayatissayisi',  
'yogunbakimatoplamyatissuresigun','serviseyatissayisi','servisetoplamyatıssüresıgun',   
'kanbasincisistolik','kanbasincidiastolik','nabiz','solunumsayisi','FEV1%','PEF%',   
'FEV1/FVC_Degeri','tanisure_ay','acilservisetoplamyatissuresi_saat','vki']  

data[numerik_degiskenler] = data[numerik_degiskenler].apply(pd.to_numeric, errors='coerce') 

data.loc[[97,105,111,112,126,131,142,154,170,181,206,210,216,223,262,272,280,298,304,320,324,329,344,356,361,375,384,389,402,410,417,422,432,457,465,490,496], 'ailedekoahveyaastimtanilihastavarmi'] = 1 
data.loc[[98,108,114,129,133,148,158,172,182,193,208,214,217,229,264,275,283,301,306,326,335,348,364,379,386,392,404,411,419,423,445,460,466,495], 'ailedekoahveyaastimtanilihastavarmi'] = 2  

kategorik_degiskenler = ['cinsiyet', 'egitimduzeyi', 'meslegi', 'sigarakullanimi', 'tani','hastaneyeyattimi','ailedekoahveyaastimtanilihastavarmi','varsakimdeanne','varsakimdebaba','varsakimdekardes','varsakimdiger','tanisure_ay'] 

data['tani'] = data['tani'].astype('category')
data['cinsiyet'] = data['cinsiyet'].astype('category')
data['egitimduzeyi'] = data['egitimduzeyi'].astype('category')
data['meslegi'] = data['meslegi'].astype('category')
data['sigarakullanimi'] = data['sigarakullanimi'].astype('category')
data['hastaneyeyattimi'] = data['hastaneyeyattimi'].astype('category')
data['ailedekoahveyaastimtanilihastavarmi'] = data['ailedekoahveyaastimtanilihastavarmi'].astype('category')
data['varsakimdeanne'] = data['varsakimdeanne'].astype('category')
data['varsakimdebaba'] = data['varsakimdebaba'].astype('category')
data['varsakimdekardes'] = data['varsakimdekardes'].astype('category')
data['varsakimdiger'] = data['varsakimdiger'].astype('category')

#Bu değişkenlerdeki NaN verileri 0 ile dolduralım. 
sifir_doldurulacak = [  
    'sigarayadevamedengundekacadeticiyo',  
    'sigarayibirakannekadargunicmis',  
    'sigarabirakangundekacadeticmis',  
    'nezamanbirakmisgun',  
    'varsakimdeanne',   
    'varsakimdebaba',   
    'varsakimdekardes',   
    'varsakimdiger',   
]  
data[sifir_doldurulacak] = data[sifir_doldurulacak].fillna(0)  

# Bu değişkenlerdeki Nan verilerini medyan ile dolduralım.
data['tanisure_ay'] = data['tanisure_ay'].fillna(data['tanisure_ay'].median())  
data['kanbasincisistolik'] = data['kanbasincisistolik'].fillna(data['kanbasincisistolik'].median())  
data['kanbasincidiastolik'] = data['kanbasincidiastolik'].fillna(data['kanbasincidiastolik'].median())  
data['PEF%'] = data['PEF%'].fillna(data['PEF%'].median())  

# Aykırı değerleri bulalım ve indekslerini yazalım
for column in numerik_degiskenler:  
    # Box plot oluşturma  
    plt.figure(figsize=(10, 6))  
    sns.boxplot(y=data[column])  
    plt.title(f'{column} için Box Plot')  
    plt.ylabel('Değerler')  
    plt.xlabel(column)  
    plt.show()  
    
    # IQR hesaplama  
    Q1 = data[column].quantile(0.25)  
    Q3 = data[column].quantile(0.75)  
    IQR = Q3 - Q1  

    # Aykırı değerlerin alt ve üst sınırları  
    lower_bound = Q1 - 1.5 * IQR  
    upper_bound = Q3 + 1.5 * IQR  

    # Aykırı değerlerin indekslerini alma  
    aykiri_degerler = data[(data[column] < lower_bound) | (data[column] > upper_bound)]  
    aykiri_indeksler = aykiri_degerler.index.tolist()  

    print(f"\n{column} için Aykırı Değerler:")  
    print(aykiri_degerler)  
    print(f"\n{column} için Aykırı Değerlerin Gözlem İndeksleri:")  
    print(aykiri_indeksler)
    
# Aykırı değerleri Winsorization edelim.
def winsorize_column(data, column, lower_quantile=0.10, upper_quantile=0.90):  
    lower_limit = data[column].quantile(lower_quantile)  
    upper_limit = data[column].quantile(upper_quantile)  
    data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)  

numerik_aykırı= ['sigarayibirakannekadargunicmis','sigarabirakangundekacadeticmis',   
'nezamanbirakmisgun','sigarayadevamedengundekacadeticiyo','acilserviseyatissayisi','yogunbakimayatissayisi',  
'yogunbakimatoplamyatissuresigun','serviseyatissayisi','servisetoplamyatıssüresıgun','kanbasincisistolik','kanbasincidiastolik','nabiz','solunumsayisi','FEV1%','PEF%','tanisure_ay','acilservisetoplamyatissuresi_saat','vki']  

for column in numerik_aykırı:  
    winsorize_column(data, column)  
    
#Model için bağımsız değişkenlerin belirlenmesi
data['tani'] = data['tani'].replace(2, 0)  # 2'yi 0 ile değiştir. 0:KOAH 1:Astım
#bağımlı değişkeni(hedef) bağımsız değişkenlerden ayıralım.
data2=data.drop(columns=['tani'])  
#veri setindeki kategorik değişkenlerin düzeylerine ayıralım.
data_one_hot = pd.get_dummies(data2, columns=['cinsiyet', 'egitimduzeyi', 'meslegi', 'sigarakullanimi','hastaneyeyattimi','ailedekoahveyaastimtanilihastavarmi','varsakimdeanne','varsakimdebaba','varsakimdekardes','varsakimdiger'],drop_first=True) 

X = data_one_hot #bağımsız değişkenler
Y = data.tani #bağımlı değişken


from statsmodels.stats.outliers_influence import variance_inflation_factor  
from statsmodels.tools.tools import add_constant  
# VIF hesaplama fonksiyonu-->varyans şişme miktarı ile çoklu bağlantı kontrolü yapalım.
# Kategorik değişkenleri çıkaralım 
X = data_one_hot.select_dtypes(exclude=['bool'])  # Boolean (True/False) olanları çıkar  
def calculate_vif(X):  
    X_with_const = add_constant(X)  # Sabit terim ekleme  
    vif_data = pd.DataFrame()  
    vif_data["feature"] = X_with_const.columns  
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]  
    return vif_data  

vif_results = calculate_vif(X)  
print(vif_results)  

# Korelasyon matrisine bakarak bağımsız değişkenler arasındaki ilişkileri inceleyelim
correlation_matrix =X.corr()  
print(correlation_matrix)
# Isı haritası ile görselleştirme  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  
plt.show()
 
for column in correlation_matrix.columns:  
    print(f"Korelasyon Değerleri için {column}:")  
    print(correlation_matrix[column].to_string(index=True))  
    print("\n")

# Isı haritası görselleştirmesi  
plt.figure(figsize=(12, 10))  # Grafik boyutunu artır  
sns.heatmap(correlation_matrix,   
            annot=True,  # Değerleri göster  
            cmap='coolwarm',  # Renk haritası  
            fmt='.2f',  # Değerleri 2 ondalık basamağa yuvarla  
            annot_kws={'size': 8},  # Yazı boyutunu artır  
            square=True)  # Kare hücreler  

# Grafik başlığı ve etiketler  
plt.title('Korelasyon Matrisi', pad=20, fontsize=14)  
plt.xticks(rotation=45, ha='right')  # X ekseni etiketlerini döndür  
plt.yticks(rotation=0)  

# Grafik düzenini ayarla  
plt.tight_layout()  
plt.show()

#VIF değerleri yüksek çıkan değişkenler çoklu bağlantı sorunu yaşatacağı için çıkaralım
data_one_hot=data_one_hot.drop(columns=['yogunbakimayatissayisi','yogunbakimatoplamyatissuresigun','serviseyatissayisi','servisetoplamyatıssüresıgun','acilservisetoplamyatissuresi_saat'])

X = data_one_hot
Y = data.tani     

# Veriyi eğitim ve test setlerine ayırma  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  

# Veriyi ölçeklendirme  
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  

# Elastik Lojistik Regresyon Modelini Kurma  
model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.3, max_iter=1000)  
model.fit(X_train, Y_train)  

# Test seti ile tahmin yapma  
Y_pred = model.predict(X_test)  

print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))

import pandas as pd  
from sklearn.model_selection import train_test_split  
import statsmodels.api as sm  

# statsmodels ile modelin özet bilgilerini alma  
X_train_sm = sm.add_constant(X_train)  # Sabit terim ekleme  
logit_model = sm.Logit(Y_train, X_train_sm).fit()  # Lojistik regresyon modeli  
print(logit_model.summary())  # Modelin özet bilgilerini yazdırma 
import statsmodels.api as sm  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import log_loss  
 
result = logit_model 

# AIC ve BIC değerlerini yazdırma  
print("AIC:", result.aic)  
print("BIC:", result.bic) 

from sklearn.metrics import roc_curve, auc  

# Test seti üzerinde tahmin olasılıklarını alma  
Y_prob = model.predict_proba(X_test)[:, 1]  

# ROC eğrisi hesaplama  
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)  
roc_auc = auc(fpr, tpr)  

# ROC eğrisini çizme  
plt.figure()  
plt.plot(fpr, tpr, color='blue', label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')  
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('Yanlış Pozitif Oranı')  
plt.ylabel('Doğru Pozitif Oranı')  
plt.title('Receiver Operating Characteristic (ROC) Eğrisi')  
plt.legend(loc='lower right')  
plt.show()

# Confusion matrix'i yazdırma  
conf_matrix = confusion_matrix(Y_test, Y_pred)  
print("Confusion Matrix:")  
print(conf_matrix)  

# Confusion matrix'i görselleştirme  
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['KOAH', 'Astım'], yticklabels=['KOAH', 'Astım'])  
plt.ylabel('Gerçek Değerler')  
plt.xlabel('Tahmin Edilen Değerler')  
plt.title('Confusion Matrix')  
plt.show()

# Eğitim seti üzerindeki tahmin  
y_train_pred = model.predict(X_train)  

# Eğitim seti performansı  
print("Eğitim Seti Performansı:")  
print("Doğruluk:", accuracy_score(Y_train, y_train_pred))  
print(classification_report(Y_train, y_train_pred))  
print("Confusion Matrix:\n", confusion_matrix(Y_train, y_train_pred))  

# Test seti performansı  
print("\nTest Seti Performansı:")  
print("Doğruluk:", accuracy_score(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))  
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Eğitim ve test performanslarını hesaplayalım  
# Eğitim seti için metrikler  
y_train_pred = model.predict(X_train)  
y_train_prob = model.predict_proba(X_train)[:, 1]  
train_accuracy = accuracy_score(Y_train, y_train_pred)  
train_auc = roc_auc_score(Y_train, y_train_prob)  

# Test seti için metrikler  
y_test_pred = model.predict(X_test)  
y_test_prob = model.predict_proba(X_test)[:, 1]  
test_accuracy = accuracy_score(Y_test, y_test_pred)  
test_auc = roc_auc_score(Y_test, y_test_prob)  

# Performans farklarını hesaplayalım  
accuracy_diff = abs(train_accuracy - test_accuracy)  
auc_diff = abs(train_auc - test_auc)  

print("PERFORMANS KARŞILAŞTIRMASI:")  
print("-" * 50)  
print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")  
print(f"Test Doğruluğu: {test_accuracy:.4f}")  
print(f"Doğruluk Farkı: {accuracy_diff:.4f}")  
print("\n")  
print(f"Eğitim AUC: {train_auc:.4f}")  
print(f"Test AUC: {test_auc:.4f}")  
print(f"AUC Farkı: {auc_diff:.4f}")  

# ROC eğrilerini karşılaştırmalı olarak çizelim  
plt.figure(figsize=(10, 6))  

# Eğitim seti ROC eğrisi  
fpr_train, tpr_train, _ = roc_curve(Y_train, y_train_prob)  
plt.plot(fpr_train, tpr_train, color='blue', label=f'Eğitim ROC (AUC = {train_auc:.3f})')  

# Test seti ROC eğrisi  
fpr_test, tpr_test, _ = roc_curve(Y_test, y_test_prob)  
plt.plot(fpr_test, tpr_test, color='red', label=f'Test ROC (AUC = {test_auc:.3f})')  

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('Yanlış Pozitif Oranı')  
plt.ylabel('Doğru Pozitif Oranı')  
plt.title('Eğitim ve Test Setleri için ROC Eğrileri Karşılaştırması')  
plt.legend(loc='lower right')  
plt.grid(True)  
plt.show()

#######################################################################################################
#######################################################################################################

#Test Veri kümesinin ön işlemesi

datatest['tanisure_ay'] = (datatest['tanisuresiyil'] * 12) + datatest['tanisuresiay']   
datatest['acilservisetoplamyatissuresi_saat']= (datatest['acilservisetoplamyatıssuresigun']* 24)+ datatest['acilservisetoplamyatissuresisaat']
datatest['boy'] = datatest['boy'] / 100  
datatest['vki'] = datatest['vucutagirligi'] / (datatest['boy'] ** 2)  

datatest = datatest.drop(columns=['hastaNo','basvurutarihi','tanisuresiyil','tanisuresiay','acilservisetoplamyatıssuresigun','acilservisetoplamyatissuresisaat','FEV1','PEF','yogunbakimatoplamyatissuresisaat','servisetoplamyatissuresisaat','boy', 'vucutagirligi'])

datatest.loc[[3,11,19,27,31,35,43,57,76,83,89], 'ailedekoahveyaastimtanilihastavarmi'] = 1 
datatest.loc[[5,13,25,28,33,38,45,59,69,79,85,92], 'ailedekoahveyaastimtanilihastavarmi'] = 2  

datatest[numerik_degiskenler] = datatest[numerik_degiskenler].apply(pd.to_numeric, errors='coerce')

# Boşlukları NaN ile değiştirme  
datatest['varsakimdebaba'] = datatest['varsakimdebaba'].replace(r'^\s*$', np.nan, regex=True)  
datatest[sifir_doldurulacak] = datatest[sifir_doldurulacak].fillna(0) 

datatest['cinsiyet'] = datatest['cinsiyet'].astype('category')
datatest['egitimduzeyi'] = datatest['egitimduzeyi'].astype('category')
datatest['meslegi'] = datatest['meslegi'].astype('category')
datatest['sigarakullanimi'] = datatest['sigarakullanimi'].astype('category')
datatest['hastaneyeyattimi'] = datatest['hastaneyeyattimi'].astype('category')
datatest['ailedekoahveyaastimtanilihastavarmi'] = datatest['ailedekoahveyaastimtanilihastavarmi'].astype('category')
datatest['varsakimdeanne'] = datatest['varsakimdeanne'].astype('category')
datatest['varsakimdebaba'] = datatest['varsakimdebaba'].astype('category')
datatest['varsakimdekardes'] = datatest['varsakimdekardes'].astype('category')
datatest['varsakimdiger'] = datatest['varsakimdiger'].astype('category')

#Test kümesindeki eksik değişkenleri eğitim kümesindeki değişkenlerin medyanları ile dolduralım 
değişkenler = ['tanisure_ay', 'kanbasincisistolik', 'kanbasincidiastolik', 'PEF%', 'yas','FEV1%','FEV1/FVC_Degeri']  

# Eğitim veri setindeki belirtilen değişkenlerin medyan değerlerini hesaplayalım 
medyan_degerleri = data[değişkenler].median()  
# Test setindeki belirtilen değişkenlere medyan değerlerini uygulayalım 
for sutun in değişkenler:  
    if sutun in datatest.columns:  
        datatest[sutun] = datatest[sutun].fillna(medyan_degerleri[sutun]) 
        
        # Kontrol için eksik değer sayısını göster  
        print("Doldurma sonrası eksik değer sayısı:")  
        print(datatest.isnull().sum())

    # Aykırı değerleri Winsorization edelim.
    def winsorize_column(datatest, column, lower_quantile=0.10, upper_quantile=0.90):  
        lower_limit = datatest[column].quantile(lower_quantile)  
        upper_limit = datatest[column].quantile(upper_quantile)  
        datatest[column] = datatest[column].clip(lower=lower_limit, upper=upper_limit)  

    numerik_aykırı= ['sigarayibirakannekadargunicmis','sigarabirakangundekacadeticmis',   
    'nezamanbirakmisgun','sigarayadevamedengundekacadeticiyo','acilserviseyatissayisi','yogunbakimayatissayisi',  
    'yogunbakimatoplamyatissuresigun','serviseyatissayisi','servisetoplamyatıssüresıgun','kanbasincisistolik','kanbasincidiastolik','nabiz','solunumsayisi','FEV1%','PEF%','tanisure_ay','acilservisetoplamyatissuresi_saat','vki']  

    for column in numerik_aykırı:  
        winsorize_column(datatest, column)  
    
model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.3, max_iter=1000)  
model.fit(X_train, Y_train)  

datatest = pd.get_dummies(datatest, columns=['cinsiyet', 'egitimduzeyi', 'meslegi', 'sigarakullanimi','hastaneyeyattimi','ailedekoahveyaastimtanilihastavarmi','varsakimdeanne','varsakimdebaba','varsakimdekardes','varsakimdiger'],drop_first=True) 
datatest=datatest.drop(columns=['yogunbakimayatissayisi','yogunbakimatoplamyatissuresigun','serviseyatissayisi','servisetoplamyatıssüresıgun','acilservisetoplamyatissuresi_saat'])

# Eğitim ve test setlerini ayırma  
X_train = X  # Eğitim seti  
Y_train = data['tani']   # Eğitim setinin hedef değişkeni  

X_test = datatest        # Test seti  
Y_test = None            # Test setinde hedef değişken yoksa None bırakabilirsiniz

# Veriyi ölçeklendirme  
from sklearn.pipeline import Pipeline  
pipeline = Pipeline([  
    ('scaler', StandardScaler()),  
    ('classifier', LogisticRegression())  
])

X_test = datatest.copy()  
X_test = scaler.fit_transform(X_test)  

# Test seti ile tahmin yapma  
Y_pred_test = model.predict(X_test)

# Eğer Y_test yoksa, confusion_matrix ve classification_report kullanamazsınız  
# Ancak tahminleri yazdırabilirsiniz  
print("Tahminler:", Y_pred_test)

# Özellik önemliliklerini görselleştirme  
feature_importance = pd.DataFrame({  
    'feature': X.columns,  
    'importance': abs(model.coef_[0])  
})  
feature_importance = feature_importance.sort_values('importance', ascending=False)

#######################################################################################################
#######################################################################################################

#Diğer Modeller

from sklearn.ensemble import (  
    RandomForestClassifier,   
    GradientBoostingClassifier,  
    ExtraTreesClassifier,  
    BaggingClassifier,  
    AdaBoostClassifier,  
    HistGradientBoostingClassifier  
)  
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.neural_network import MLPClassifier  
from sklearn.linear_model import LogisticRegression, RidgeClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  
import warnings  
warnings.filterwarnings('ignore')  

# Güncellenmiş model listesi  
models = {  
    'Random Forest': RandomForestClassifier(random_state=42),  
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),  
    'Extra Trees': ExtraTreesClassifier(random_state=42),  
    'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),  
    'AdaBoost': AdaBoostClassifier(random_state=42),  
    'Bagging': BaggingClassifier(random_state=42),  
    'SVM': SVC(probability=True, random_state=42),  
    'KNN': KNeighborsClassifier(),  
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000),  
    'Logistic Regression': LogisticRegression(random_state=42),  
    'Ridge Classifier': RidgeClassifier(random_state=42),  
    'Decision Tree': DecisionTreeClassifier(random_state=42),  
    'Gaussian NB': GaussianNB(),  
    'LDA': LinearDiscriminantAnalysis(),  
    'QDA': QuadraticDiscriminantAnalysis()  
}  

# Performans metriklerini hesaplama  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  

results = {}  
for name, model in models.items():  
    print(f"\nEğitiliyor: {name}")  
    
    # Model eğitimi  
    model.fit(X_train, Y_train)  
    
    # Tahminler  
    train_pred = model.predict(X_train)  
    test_pred = model.predict(X_test)  
    
    # Olasılık tahminleri (ROC AUC için)  
    if hasattr(model, "predict_proba"):  
        train_prob = model.predict_proba(X_train)[:, 1]  
        test_prob = model.predict_proba(X_test)[:, 1]  
    else:  
        train_prob = train_pred  
        test_prob = test_pred  
    
    # Metrikleri hesaplama  
    results[name] = {  
        'Train Accuracy': accuracy_score(Y_train, train_pred),  
        'Test Accuracy': accuracy_score(Y_test, test_pred),  
        'Train Precision': precision_score(Y_train, train_pred),  
        'Test Precision': precision_score(Y_test, test_pred),  
        'Train Recall': recall_score(Y_train, train_pred),  
        'Test Recall': recall_score(Y_test, test_pred),  
        'Train F1': f1_score(Y_train, train_pred),  
        'Test F1': f1_score(Y_test, test_pred),  
        'Train ROC AUC': roc_auc_score(Y_train, train_prob),  
        'Test ROC AUC': roc_auc_score(Y_test, test_prob)  
    }  

# Sonuçları DataFrame'e dönüştürme  
results_df = pd.DataFrame(results).T  

# Sonuçları görselleştirme  
import matplotlib.pyplot as plt  
import seaborn as sns  

# Test metriklerini görselleştirme  
plt.figure(figsize=(15, 8))  
test_metrics = results_df[[col for col in results_df.columns if 'Test' in col]]  
sns.heatmap(test_metrics, annot=True, cmap='YlOrRd', fmt='.3f')  
plt.title('Test Performans Metrikleri Karşılaştırması')  
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()  

# En iyi modelleri belirleme  
print("\nTest Accuracy'e göre en iyi 5 model:")  
print(results_df.sort_values('Test Accuracy', ascending=False)[['Test Accuracy', 'Test F1', 'Test ROC AUC']].head())  

# Overfitting analizi  
plt.figure(figsize=(12, 6))  
models_list = results_df.index  
train_acc = results_df['Train Accuracy']  
test_acc = results_df['Test Accuracy']  

x = range(len(models_list))  
width = 0.35  

plt.bar([i - width/2 for i in x], train_acc, width, label='Train Accuracy', color='skyblue')  
plt.bar([i + width/2 for i in x], test_acc, width, label='Test Accuracy', color='lightcoral')  

plt.xlabel('Models')  
plt.ylabel('Accuracy')  
plt.title('Train vs Test Accuracy Comparison')  
plt.xticks(x, models_list, rotation=45, ha='right')  
plt.legend()  
plt.tight_layout()  
plt.show()

# Eğitim ve Test sonuçlarını yan yana gösterme  
comparison_df = pd.DataFrame({  
    'Train Accuracy': results_df['Train Accuracy'],  
    'Test Accuracy': results_df['Test Accuracy'],  
    'Fark': results_df['Train Accuracy'] - results_df['Test Accuracy']  
}).sort_values('Test Accuracy', ascending=False)  

# Sonuçları yüzde formatında gösterme  
comparison_df = comparison_df.round(4) * 100  

print("\nEğitim ve Test Doğruluk Oranları Karşılaştırması:")  
print("=" * 60)  
print(comparison_df)  

# Overfitting analizi  
print("\nOverfitting Analizi:")  
print("=" * 60)  
for model in comparison_df.index:  
    diff = comparison_df.loc[model, 'Fark']  
    if diff > 5:  # 5 puandan fazla fark varsa  
        print(f"{model}: Yüksek overfitting riski! (Fark: {diff:.2f} puan)")  
    elif diff < 2:  # 2 puandan az fark varsa  
        print(f"{model}: İyi genelleme! (Fark: {diff:.2f} puan)")  
    else:  
        print(f"{model}: Normal fark (Fark: {diff:.2f} puan)")  

# Görselleştirme  
plt.figure(figsize=(12, 6))  
comparison_df[['Train Accuracy', 'Test Accuracy']].plot(kind='bar', figsize=(12, 6))  
plt.title('Eğitim vs Test Doğruluk Oranları')  
plt.xlabel('Model')  
plt.ylabel('Doğruluk Oranı (%)')  
plt.legend(loc='lower right')  
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  
plt.show()

###############################################################################################################