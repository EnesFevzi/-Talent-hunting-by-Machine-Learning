######################
#İş Problemi
######################
#Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
#(average, highlighted) oyuncu olduğunu tahminleme.

#####################
#Veri Seti Hikayesi
#####################
#Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
#içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.



###########################
#scoutium_attributes.csv
############################

#task_response_id: :Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id          :İlgili maçın id'si
#evaluator_id      :Değerlendiricinin(scout'un) id'si
#player_id         :İlgili oyuncunun id'si
#position_id       :İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#1: Kaleci
#2: Stoper
#3: Sağ bek
#4: Sol bek
#5: Defansif orta saha
#6: Merkez orta saha
#7: Sağ kanat
#8: Sol kanat
#9: Ofansif orta saha
#10: Forvet
#analysis_id        :Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
#attribute_id       :Oyuncuların değerlendirildiği her bir özelliğin id'si
#attribute_value    :Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)


################################
#scoutium_potential_labels.csv
################################

#task_response_id    :Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id            :İlgili maçın id'si
#evaluator_id        :Değerlendiricinin(scout'un) id'si
#player_id           :İlgili oyuncunun id'si
#potential_label     :Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from matplotlib import rc,rcParams
import itertools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler





#Bazı uyarıları almamak adına görmezden geliyoruz.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


sc_attr_ =pd.read_csv("C:/Users/Enes Fevzi/Desktop/Machine Learning/Case Study II/Scoutium-220805-075951/scoutium_attributes.csv",sep=";")
sc_attr = sc_attr_.copy()
sc_attr.head(5)


sc_attr.shape
sc_attr.columns



sc_pot_lab_ =pd.read_csv("C:/Users/Enes Fevzi/Desktop/Machine Learning/Case Study II/Scoutium-220805-075951/scoutium_potential_labels.csv", sep=";")
sc_pot_lab= sc_pot_lab_.copy()
sc_pot_lab.head(5)

sc_pot_lab.shape
sc_pot_lab.columns


df =pd.merge(sc_attr,sc_pot_lab,on=["task_response_id","match_id","evaluator_id", "player_id"])
df.head(5)


#Eksik değerler
df.isnull().sum()
"""task_response_id    0
match_id            0
evaluator_id        0
player_id           0
position_id         0
analysis_id         0
attribute_id        0
attribute_value     0
potential_label     0
dtype: int64"""

#Her sütunun veri türü.
df.dtypes
"""task_response_id      int64
match_id              int64
evaluator_id          int64
player_id             int64
position_id           int64
analysis_id           int64
attribute_id          int64
attribute_value     float64
potential_label      object
dtype: object
"""

#Dizi boyutlarının uzunlukların
df.shape
"""(10730, 9)"""

#Sondan doğru bakmak için.
df.tail()
"""       task_response_id  match_id  evaluator_id  player_id  position_id  analysis_id  attribute_id  attribute_value potential_label
10725              5642     63032        151191    1909728            7     12825756          4357           67.000     highlighted
10726              5642     63032        151191    1909728            7     12825756          4407           78.000     highlighted
10727              5642     63032        151191    1909728            7     12825756          4408           67.000     highlighted
10728              5642     63032        151191    1909728            7     12825756          4423           67.000     highlighted
10729              5642     63032        151191    1909728            7     12825756          4426           78.000     highlighted"""


df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T


df.head(5)
df.info()



###############################################################################
# position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırıyoruz.
#Kuracağımız sistemde kalecilerin olmasını istemiyoruz.
###############################################################################

df = df.loc[df["position_id"]!=1]

df.groupby("position_id").count()

################################################################################
# potential_label içerisindeki below_average sınıfını veri setinden kaldırıyoruz.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur).
###############################################################################

df = df.loc[df["potential_label"]!="below_average"]#veya #df = df[~df["potential_label"].str.contains("below_average", na=False)]

df.index


#################################################################################
# Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturuyoruz.
#################################################################################
#Oluşturduğumuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturuyoruz. Bu pivot table'da her satırda bir oyuncu
#olacak şekilde manipülasyon yapıyoruz.
#İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#attribute_value” olacak şekilde pivot table’ı oluşturuyoruz.

df_pivot= df.pivot_table(values="attribute_value",index=["player_id", "position_id", "potential_label"],columns="attribute_id")


#“reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayoruz ve “attribute_id” sütunlarının isimlerini stringe çeviriyoruz.


df_pivot.reset_index(inplace=True)

df = df_pivot

df.head()
##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################
#Bu fonksiyon  göz ile yapamayacağımız olayları yapmamızı sağlar.Çok önemli fonksiyondur.İş hayatımızda hep lazım olabilir.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)
"""Observations: 271
Variables: 37
cat_cols: 7
num_cols: 30
cat_but_car: 0
num_but_cat: 6"""


##################
# Label Encoding #Değişkenlerin tamsil şekillerininin değiştirilmesi
##################
# Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediyoruz.

#label enc. kategorik değişkenleri işlemek için.
#label encoding / binary encoding işlemini 2 sınıflı kategorik değişkenlere uyguluyoruz. bu iki sınıfı 1-0 şeklinde encodelamış oluyoruz.
#one-hot encoder ise ordinal sınıflı kategorik değişkenler için uyguluyoruz. sınıfları arasında fark olan
#değişkenleri sınıf sayısınca numaralandırıp kategorik değişken olarak df e gönderiyor.

le = LabelEncoder()

le.fit(df["potential_label"])
df["potential_label"] = le.transform(df["potential_label"])
df.head()

df.groupby("potential_label").count()

"""
def label_encoder(df, potential_label):
    labelencoder = LabelEncoder()
    df[potential_label] = labelencoder.fit_transform(df[potential_label])
    return df

potential_label = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in potential_label:
    label_encoder(df, col)

df.groupby("potential_label").count()

df.head()
"""




################################################################################
# Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atıyoruz.
################################################################################


num_cols = [col for col in df.columns if col not in ["player_id", "position_id", "potential_label"]]

df.shape
df.head()



######################################
# Korelasyon Analizi (Analysis of Correlation)
######################################
#(-1,1) arası değer alır.0 değeri ilişki yok anlamındadır.

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

##################################
# STANDARTLAŞTIRMA
# Değişken (özellik) sütunlarının ortalama değeri 0 ve standart sapması 1 olacak şekilde standart normal dağılım oluşturmaktır.
##################################
# Kaydettiğimiz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için Standard Scaler uyguluyoruz.

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


"""
##################################
# BASE MODELLEME
##################################
#Y BAĞIMLI, X BAĞIMSIZ

y = df["potential_label"]
X = df.drop(["potential_label", "player_id"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# Auc: 0.7282

"""

##################################
# MODELLEME
##################################
#Y BAĞIMLI, X BAĞIMSIZ

y = df["potential_label"]
X = df.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 0.3753 (LR) 
RMSE: 0.3725 (Ridge) 
RMSE: 0.4015 (Lasso) 
RMSE: 0.4015 (ElasticNet) 
RMSE: 0.3587 (KNN) 
RMSE: 0.4105 (CART) 
RMSE: 0.3063 (RF) 
RMSE: 0.313 (SVR) 
RMSE: 0.3183 (GBM) 
RMSE: 0.3412 (XGBoost) 
RMSE: 0.3242 (LightGBM) 
RMSE: 0.3094 (CatBoost) 
"""
# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


################################################
# Random Forests
################################################
#Birden çok karar ağacı üzerinden her bir karar ağacını farklı bir gözlem örneği
#üzerinde eğiterek çeşitli modeller üretip, sınıflandırma oluşturmanızı sağlamaktadır.

rf_model = RandomForestClassifier(random_state=17) #Model nesnemizi getirdik.
rf_model.get_params() #Bu modelin hiperparametrelerine bakıyoruz.


"""{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': None,
 'oob_score': False,
 'random_state': 17,
 'verbose': 0,
 'warm_start': False}
"""

#Hipermarametre optimizasyonu yapmadan önce bi gözlemliyoruz...
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
cv_results['test_accuracy'].mean()
#0.8707671957671957

cv_results['test_f1'].mean()
#0.5667748917748918

cv_results['test_roc_auc'].mean()
# 0.8983766233766234

cv_results["test_precision"].mean()
# 0.905

cv_results["test_recall"].mean()
#0.43666666666666665


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"], #Bölünlemelerde göz önünde bulundurulması gereken değişken sayısı
             "min_samples_split": [2, 5, 8, 15, 20],#Kaç tane gözlem birimi olacağını belirler.
             "n_estimators": [100, 200, 500]}  #Fit edilecek bağımsız ağaç sayısı....

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y) #5 katlı cv
#Fitting 5 folds for each of 180 candidates, totalling 900 fits



rf_best_grid.best_params_
"""{'max_depth': None,
 'max_features': 5,
 'min_samples_split': 5,
 'n_estimators': 200}"""

rf_best_grid.best_score_
#0.8891582491582491

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y) #en iyi parametreler..


cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

cv_results['test_accuracy'].mean()
#0.8818783068783068
cv_results['test_f1'].mean()
#0.5996031746031745
cv_results['test_roc_auc'].mean()
#0.9087806637806638
cv_results["test_precision"].mean()
#0.9333333333333333
cv_results["test_recall"].mean()
#0.47333333333333333


################################################
# GBM Gradient Boosting Machines (Zayıf sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturması fikrine dayanır.
################################################
#Hatalar/artıklar üzerine tek bir tahminsel model formunda olan modeller serisi kurulur.

gbm_model = GradientBoostingClassifier(random_state=17)
#ön tanımlı parametreler
gbm_model.get_params()
"""{'ccp_alpha': 0.0,
 'criterion': 'friedman_mse',
 'init': None,
 'learning_rate': 0.1,
 'loss': 'deviance',
 'max_depth': 3,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_iter_no_change': None,
 'random_state': 17,
 'subsample': 1.0,
 'tol': 0.0001,
 'validation_fraction': 0.1,
 'verbose': 0,
 'warm_start': False}
"""
#learning_rate': 0.1, öğrenme oranı / n_estimators': 100 optimizasyon sayısı

#Hatalarımız nelerdir onlara bakıyoruz.
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
cv_results['test_accuracy'].mean()
#0.8671380471380472

cv_results['test_f1'].mean()
#0.6097580552688913

cv_results['test_roc_auc'].mean()
#0.8706483439041579

cv_results["test_precision"].mean()
#0.7766666666666666

cv_results["test_recall"].mean()
#0.5303030303030304

gbm_params = {"learning_rate": [0.01, 0.1], #ne kadar küçük olursa train süresi uzamasıdır.Ama train başarısı artmaktadır.
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
"""{'learning_rate': 0.01,
 'max_depth': 10,
 'n_estimators': 1000,
 'subsample': 0.5}"""

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

cv_results['test_accuracy'].mean()
#0.8782491582491584

cv_results['test_f1'].mean()
#0.6102698412698413

cv_results['test_roc_auc'].mean()
#0.9008809020436926

cv_results["test_precision"].mean()
#0.8984615384615384

cv_results["test_recall"].mean()
#0.4954545454545454

################################################
# XGBoost
################################################
#GBM nin hız ve tahmin pefonmanasını arttırmak için optimize edilmiş;ölçeklenebilir ve farklı platformlara entegre edilebilir versiyonudur.

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()

cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()


xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]} #subsample değişkenlerden alınann gözlem sayılarıyla ilişkiliymiş

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)



cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()

################################################
# LightGBM
################################################
# XGboost un eğitim süresi perfonmansını arttırmaya yönelik geliştirilen bir diğer GBM türüdür.Leaf wise büyütme stratejisi ile daha hızlıdır.
#Doğasında aşırı öğrenmenin önüne geçmek vardır.

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()


################################################
# CatBoost #Kategorik değişkenler ile otomatik olarak mücadele edebilen hızlı başarılı bir diğer GBM türevi.
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]













