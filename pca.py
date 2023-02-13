# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class DimensionalityReductionPCA:

    def __init__(self, vec):
      self.vec = vec
      self.dim = None
      self.shape_before = None
      self.shape_after = None
      self.x_pca = None
      self.pca = None


    def _reduce(self):
      """
      Work out number of dimensions that preserves 95% variance
      """
      pca = PCA()
      vector = self.vec.toarray()
      pca.fit(vector)
      cumsum = np.cumsum(pca.explained_variance_ratio_)
      self.dim = np.argmax(cumsum >= 0.95) + 1

      print(f'Number of dimensions: {self.dim}')

      self.pca = PCA(n_components=self.dim)
      self.x_pca = self.pca.fit_transform(vector)
      self.shape_before = vector.shape
      self.shape_after = self.x_pca.shape
      

    def _explained_variance_ratio_(self):
      return self.pca.explained_variance_ratio_

    def plot(self):
      PC_values = np.arange(self.pca.n_components_) + 1
      plt.plot(PC_values, self.pca.explained_variance_ratio_, 'ro-', 
              linewidth=2)
      plt.title('Scree Plot')
      plt.xlabel('Principal Component')
      plt.ylabel('Proportion of Variance Explained')
      plt.show()

      plt.plot(range(dim), self.pca.explained_variance_ratio_)
      plt.plot(range(dim), np.cumsum(pca.explained_variance_ratio_))
      plt.title("Component-wise and Cumulative Explained Variance")
      plt.show()

