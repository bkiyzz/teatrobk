from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def gauss_pdf(mu, sigma, x):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

iris = datasets.load_iris()

X = iris.data
y = iris.target

# 0 - setosa
# 1 - virginica
# 2 - versicolor

x_sep_len = X[:, 0]

# _, bins, _ = plt.hist(x_sep_len, bins = 20)
# plt.show()

##### *********** Sepal length distributions *********** #####
x_sep_len_set = X[:50, 0]
x_sep_len_vir = X[50:100, 0]
x_sep_len_ver = X[100:, 0]

plt.hist([x_sep_len_set, x_sep_len_vir, x_sep_len_ver], bins = 20)
plt.legend(["Setosa", "Virginica", "Versicolor"])
plt.xlabel("Sepal length (cm)")
plt.ylabel("Number of flowers")
plt.title("Distributions of the sepal length")

x_sep_len_set_mean = np.mean(x_sep_len_set)
x_sep_len_vir_mean = np.mean(x_sep_len_vir)
x_sep_len_ver_mean = np.mean(x_sep_len_ver)

x_sep_len_set_std = np.std(x_sep_len_set)
x_sep_len_vir_std = np.std(x_sep_len_vir)
x_sep_len_ver_std = np.std(x_sep_len_ver)

x = np.arange(3.5, 9, 0.01)

x_sep_len_set_norm = gauss_pdf(x_sep_len_set_mean, x_sep_len_set_std, x)
x_sep_len_vir_norm = gauss_pdf(x_sep_len_vir_mean, x_sep_len_vir_std, x)
x_sep_len_ver_norm = gauss_pdf(x_sep_len_ver_mean, x_sep_len_ver_std, x)

plt.plot(x, x_sep_len_set_norm * 10, color = "blue")
plt.plot(x, x_sep_len_vir_norm * 10, color = "orange")
plt.plot(x, x_sep_len_ver_norm * 10, color = "green")
plt.show()

norma = np.linalg.norm([x_sep_len_set_norm, x_sep_len_vir_norm, x_sep_len_ver_norm])

plt.plot(x, x_sep_len_set_norm/norma, color = "blue")
plt.plot(x, x_sep_len_vir_norm/norma, color = "orange")
plt.plot(x, x_sep_len_ver_norm/norma, color = "green")
plt.xlabel("Sepal length (cm)")
plt.ylabel("f(x)")
plt.title("Normalized original distributions")
plt.legend(["Setosa", "Virginica", "Versicolor"])
plt.show()
        
