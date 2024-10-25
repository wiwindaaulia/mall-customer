import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace=True)

x = df.drop(['CustomerID',	'Gender'], axis=1)

st.header("isi dataset")
st.write(x)

# menampilkan panah elbow
cluster = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(x)
    cluster.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=cluster, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

#panah elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data', arrowprops=dict(arrowstyle='->',
            connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data', arrowprops=dict(arrowstyle='->',
            connectionstyle='arc3', color='blue', lw=2))
st.pyplot(fig)

# Sidebar untuk memilih jumlah cluster
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster:", 2, 10, 3, 1)

# Fungsi untuk menjalankan KMeans dan menampilkan scatter plot
def k_means(n_clust):
    # Menjalankan KMeans
    kmeans = KMeans(n_clusters=n_clust).fit(x)
    x['labels'] = kmeans.labels_

    # Membuat figure dan axes secara eksplisit
    fig, ax = plt.subplots(figsize=(10, 8))

    # Membuat scatter plot dengan hue berdasarkan label cluster
    sns.scatterplot(x=x['Income'], y=x['Score'], hue=x['labels'], 
                    palette=sns.color_palette('hls', n_clust), ax=ax)

    # Menambahkan anotasi ke setiap cluster
    for label in x['labels'].unique():
        ax.annotate(label,
                    (x[x['labels'] == label]['Income'].mean(),
                     x[x['labels'] == label]['Score'].mean()),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', size=10, weight='bold', color='black')

    # Menampilkan plot dan tabel hasil clustering di Streamlit
    st.header('Cluster Plot')
    st.pyplot(fig)  # Menampilkan fig di Streamlit
    st.write(x)     # Menampilkan DataFrame dengan label cluster

# Memanggil fungsi k_means dengan nilai clust dari slider
k_means(clust)