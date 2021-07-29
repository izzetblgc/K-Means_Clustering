import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_= pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

df.shape
df.describe().T

df.isnull().sum()
df.dropna(inplace=True)
df.shape

# İade edilen ürünlerin çıkarılması
df = df[~ df["Invoice"].str.contains("C",na=False)]

df["TotalPrice"] = df["Quantity"] * df["Price"]

## RFM metriklerinin oluşturulması

df["InvoiceDate"].max()

today_date = dt.datetime(2011,12,11)

#recency: Müşterinin son satın alımından itibaren geçen süre
#frequency: Sıklık
#monetary: Müşterinin yaptığı toplam harcama

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,
                                    "Invoice": lambda y: y.nunique(),
                                    "TotalPrice": lambda z: z.sum()})

rfm.columns = ["recency","frequency","monetary"]
rfm = rfm[rfm["monetary"]>0]

pd.options.mode.chained_assignment = None

rfm["recency_score"] = pd.qcut(rfm["recency"], 5 , labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str))
rfm.head()


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
rfm = rfm[["recency", "frequency", "monetary", "segment"]]


###############################################################
# K-Means Clustering
###############################################################

# Min - Max Scaler
scaler = MinMaxScaler()
segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])
segment_data.head()

################################
# Optimum Küme Sayısının Belirlenmesi - Automatic
################################

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(segment_data)
elbow.show()

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(segment_data) # n_clusters = 6
segment_data["clusters"] = kmeans.labels_
print(f"Number of cluster selected: {elbow.elbow_value_}") # 6


################################
# RFM ve K-Means Clusterlarının Birleştirilmesi
################################
segmentation = rfm.merge(segment_data, on="Customer ID")
seg_desc = segmentation[["segment", "clusters", "recency", "frequency", "monetary"]].groupby(["clusters", "segment"]).agg(["mean", "count"])

print(seg_desc)

#                              recency       frequency        monetary
#                                 mean count      mean count      mean count
# clusters segment
# 0        about_to_sleep        38.14    78      1.21    78    493.31    78
#          champions              6.45   613     10.30   613   4303.58   613
#          loyal_customers       26.10   618      6.71   618   2746.28   618
#          need_attention        38.12    51      2.33    51    971.55    51
#          new_customers          7.43    42      1.00    42    388.21    42
#          potential_loyalists   17.43   483      2.01   483    694.57   483
#          promising             23.51    94      1.00    94    294.01    94
# 1        at_Risk              321.88    32      2.62    32    687.37    32
#          cant_loose           329.33     3     15.67     3   2163.91     3
#          hibernating          334.35   280      1.04   280    610.76   280
# 2        at_Risk              152.31   229      2.83   229   1027.16   229
#          cant_loose           148.86    22      7.64    22   2706.16    22
#          hibernating          153.77   256      1.14   256    426.92   256
# 3        about_to_sleep        57.63   274      1.15   274    465.93   274
#          at_Risk               89.04   217      3.19   217   1202.07   217
#          cant_loose            88.91    33      8.18    33   2811.60    33
#          hibernating           87.10   195      1.17   195    565.48   195
#          loyal_customers       57.13   198      5.19   198   2060.56   198
#          need_attention        57.79   136      2.32   136    869.91   136
# 4        at_Risk              232.14   115      2.46   115   1087.51   115
#          cant_loose           236.00     5      8.60     5   3469.54     5
#          hibernating          244.37   340      1.08   340    390.48   340
# 5        champions              3.60    20     77.45    20  85149.97    20
#          loyal_customers       27.33     3     45.00     3  80207.92     3
#          potential_loyalists    1.00     1      2.00     1 168472.50     1

