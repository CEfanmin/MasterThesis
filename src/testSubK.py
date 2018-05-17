import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
sns.set(font_scale=1.5)
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


if __name__ =="__main__":
    # load exo_data
    def loadData(file_name):
        exo_sample_data = pd.read_csv(file_name)
        n_features = exo_sample_data.shape[1]
        n_cluster=4
        from sklearn.preprocessing import StandardScaler
        normed_exo_data = pd.DataFrame(
            StandardScaler().fit_transform(exo_sample_data),
            columns=exo_sample_data.columns
        )
        return n_features, n_cluster,normed_exo_data

    n_features,n_clusters,normed_exo_data = loadData("../../data/0307/test1/exo_sample_data.csv")
    exo_targets = pd.read_csv("../../data/0307/test1/result01.csv")["targets"].tolist()

    from collections import Counter
    print("Counter(exo_targets) is ",Counter(exo_targets))

    # do PCA without decomposition
    from sklearn.decomposition import PCA
    full_pca = PCA(n_components=n_features, random_state=14).fit(normed_exo_data)
    # plot number of compressed features vs Accumulative Variance Ratio
    plt.plot(range(1, n_features + 1), np.cumsum(full_pca.explained_variance_ratio_))
    plt.xlabel("Number of features compressed by PCA")
    plt.ylabel("Accumulative Variance Ratio")
    plt.show()

    # do PCA with decomposition
    threshold_accum_var_ratio = 0.8
    pca_n_features = int(np.nanmin(np.where(
        np.cumsum(full_pca.explained_variance_ratio_) > threshold_accum_var_ratio, range(1, n_features + 1), np.nan
    )))
    print("pca_n_features: ", pca_n_features)
    partial_pca = PCA(n_components=pca_n_features, random_state=14)
    decomposed_pca_exo_features = pd.DataFrame(
        partial_pca.fit_transform(normed_exo_data),
        columns=['x%02d' % x for x in range(1, pca_n_features + 1)]
    )

    # Visualize decomposed PCA data with t-SNE
    pca_tsne = TSNE(n_components=2, random_state=14)
    decomposed_pca_exo_features = pd.DataFrame(
        pca_tsne.fit_transform(decomposed_pca_exo_features),
        columns=['x%02d' % x for x in range(1, 3)]
    )
    ax = None
    for c in range(n_clusters):
        ax = decomposed_pca_exo_features.iloc[
            list(np.where(np.array(exo_targets) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('exo features decomposed by PCA (Ground Truth)')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()


    # k-Means with decomposed data by PCA
    from sklearn.cluster import KMeans
    pca_km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=14)
    pca_clusters = pca_km.fit_predict(decomposed_pca_exo_features)
    ax = None
    for c in range(n_clusters):
        ax = decomposed_pca_exo_features.iloc[
            list(np.where(np.array(pca_clusters) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='cluster %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('exo features decomposed by PCA (k-Means clustering)')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()
    '''
    # NMI and AMI score
    from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
    # pca_nmi_score = normalized_mutual_info_score(exo_targets, pca_clusters)
    # pca_ami_score = adjusted_mutual_info_score(exo_targets, pca_clusters)
    # print("pca_nmi_score, pca_ami_score",pca_nmi_score, pca_ami_score)

    # using SubKmeans
    from subspace_k_means import SubspaceKMeans
    skm = SubspaceKMeans(n_clusters=n_clusters, n_jobs=-1, random_state=14)
    skm_clusters = skm.fit_predict(normed_exo_data)
    print("skm.m_", skm.m_)
    # Visualize Cluster-Space
    transformed_exo_features = pd.DataFrame(
        skm.transform(normed_exo_data),
        columns=['x%02d' % x for x in range(1, n_features + 1)]
    )
    ax = None
    for c in range(n_clusters):
        ax = transformed_exo_features.iloc[
            list(np.where(np.array(exo_targets) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('the Cluster-Space of exo features (Ground Truth)')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()

    ax = None
    for c in range(n_clusters):
        ax = transformed_exo_features.iloc[
            list(np.where(np.array(skm_clusters) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='cluster %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('the Cluster-Space of exo features (predicted clusters)')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()

    
    # NMI and AMI scores
    skm_nmi_score = normalized_mutual_info_score(exo_targets, skm_clusters)
    skm_ami_score = adjusted_mutual_info_score(exo_targets, skm_clusters)
    print("skm_nmi_score, skm_ami_score ", skm_nmi_score, skm_ami_score)

    # Visualize Noise-Space
    noise_tsne = TSNE(n_components=2, random_state=14)
    visualized_noise_wine_features = pd.DataFrame(
        noise_tsne.fit_transform(transformed_exo_features.iloc[:, skm.m_:]),
        columns=['x%02d' % x for x in range(1, 3)]
    )
    ax = None
    for c in range(n_clusters):
        ax = visualized_noise_wine_features.iloc[
            list(np.where(np.array(exo_targets) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('the Noise-Space of exo features (Ground Truth)')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()

    ax = None
    for c in range(n_clusters):
        ax = visualized_noise_wine_features.iloc[
            list(np.where(np.array(skm_clusters) == c)[0]), :
        ].plot(
            kind='scatter', x='x01', y='x02', color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('the Noise-Space of exo features (predicted clusters)')
    plt.xlabel('1st feature of the Noise-Space')
    plt.ylabel('2nd feature')
    plt.show()
    '''