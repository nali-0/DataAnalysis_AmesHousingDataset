# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## 1. Загрузка данных

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv("AmesHousing.csv")
data.drop(columns=["Order"], inplace=True)
data.drop(columns=["PID"], inplace=True)
print(f"Размерность DataFrame: {data.shape}")
data.head()

# %% [markdown]
# ## 2. Очистка пропусков
# Категориальные: заменяем NaN на "None"
# Числовые: заменяем на 0, где NaN означает отсутствие подвала/гаража
# Lot Frontage: заполняем медианой по соседству
# Очищаем другие пропуски, заполняя 0, None или модой

# %%
for column in ["Alley", "Pool QC", "Fence", "Misc Feature", "Fireplace Qu","Garage Type", "Garage Finish",
               "Garage Qual", "Garage Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]:
    data[column] = data[column].fillna("None")

# %%
for column in ["Bsmt Full Bath", "Bsmt Half Bath", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
               "Total Bsmt SF", "Garage Area", "Garage Cars"]:
    data[column] = data[column].fillna(0)

# %%
data["Lot Frontage"] = data.groupby("Neighborhood")["Lot Frontage"].transform(lambda x: x.fillna(x.median()))
data["Lot Frontage"] = data["Lot Frontage"].fillna(data["Lot Frontage"].median())

# %%
data["Mas Vnr Area"] = data["Mas Vnr Area"].fillna(0)
data["Mas Vnr Type"] = data["Mas Vnr Type"].fillna("None")
data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

for col in data.columns[data.isnull().any()]:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())

# %%
print(f"Пропусков после очистки: {data.isnull().sum().sum()}")

# %% [markdown]
# ## 3. Создание новых признаков
# 1. Возраст дома
# 2. Лет с последнего ремонта

# %%
data["Age at Sale"] = data["Yr Sold"] - data["Year Built"]
data["Years Since Remodel"] = data["Yr Sold"] - data["Year Remod/Add"]
data["Age at Sale"] = data["Age at Sale"].clip(lower=0)
data["Years Since Remodel"] = data["Years Since Remodel"].clip(lower=0)

# %% [markdown]
# ## 4. One-Hot Encoding категориальных признаков

# %%
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]
categorical_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Категориальные признаки: {categorical_cols}")

# %% [markdown]
# ## 5. Ridge-регрессия и топ 10 важных признаков по модулю

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

coef_data = pd.DataFrame({"feature": X.columns, "coef": ridge.coef_})
coef_data["abs"] = np.abs(coef_data["coef"])
top10 = coef_data.sort_values("abs", ascending=False).head(10)
print("Ridge:")
print(top10[["feature", "coef"]].to_string(index=False))

# %% [markdown]
# ## 6. Аномалии: дома с большой жилой площадью, но низкой ценой
# Используется Z-score

# %%
slope, intercept, _, _, _ = stats.linregress(data["Gr Liv Area"], data["SalePrice"])
data["residual"] = data["SalePrice"] - (slope * data["Gr Liv Area"] + intercept)

z_scores = np.abs(stats.zscore(data["residual"].dropna()))

anomalies = data[(z_scores > 3)]
print(f"Всего аномалий: {len(anomalies)}")

# %%
plt.figure(figsize=(8, 5))
plt.scatter(data["Gr Liv Area"], data["SalePrice"], alpha=0.5, label="All")
plt.scatter(anomalies["Gr Liv Area"], anomalies["SalePrice"], color="red", label="Anomalies")
plt.xlabel("Gr Liv Area")
plt.ylabel("SalePrice")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Сравнение r2 score до и после удаления аномалий

# %%
def model_score(x, y):
    X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_prediction = model.predict(X_te)
    r2 = r2_score(y_te, y_prediction)
    return r2


# %%
r2_before = model_score(X_scaled, y)
print(f"r2 до удаления: {r2_before:.3f}")

anomaly_indices = anomalies.index
X_cleared = np.delete(X_scaled, anomaly_indices, axis=0)
y_cleared = y.drop(anomaly_indices)
r2_after = model_score(X_cleared, y_cleared)
print(f"r2 после удаления: {r2_after:.3f}")
print(f"Разница в r2: {r2_after - r2_before:.3f}")

# %% [markdown]
# ## 8. Сегментация объектов недвижимости на 5 групп
# Используем KMeans

# %%
segments = ["Gr Liv Area", "Lot Area", "Overall Qual", "Overall Cond", "Year Built", "Total Bsmt SF", "Garage Area"]
X_seg = data[segments].fillna(0)
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

kmeans = KMeans(n_clusters=5, n_init=10)
data["Segment"] = kmeans.fit_predict(X_seg_scaled)
print(data["Segment"].value_counts().sort_index())

# %% [markdown]
# ## 9. PCA на всех числовых признаках + регрессия

# %%
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

Xpca_train, Xpca_test, ypca_train, ypca_test = train_test_split(X_pca, y, test_size=0.2)
pca_model = LinearRegression()
pca_model.fit(Xpca_train, ypca_train)
ypca_pred = pca_model.predict(Xpca_test)
print(f"PCA Regression: r2 = {r2_score(ypca_test, ypca_pred):.4f}")

# %% [markdown]
# ## 8. Динамика цен: годы, месяца, кризис 2008, сезонность(весна, зима)

# %%
yearly = data.groupby("Yr Sold")["SalePrice"].mean()
print("Средняя цена по годам:", yearly.round(0))
plt.plot(yearly.index, yearly.values, marker="o")
plt.title("График цен по годам")
plt.tight_layout()
plt.show()

# %%
monthly = data.groupby("Mo Sold")["SalePrice"].mean()
print("Средняя цена по месяцам:", monthly.round(0))
plt.bar(monthly.index, monthly.values)
plt.title("График цен по месяцам")
plt.tight_layout()
plt.show()

# %%
mean2007 = data[data["Yr Sold"] == 2007]["SalePrice"].mean()
mean2009 = data[data["Yr Sold"] == 2009]["SalePrice"].mean()
drop = (mean2009 - mean2007) / mean2007 * 100
print(f"Кризис 2008 {drop:.2f}%")

# %%
winter = data[data["Mo Sold"].isin([12, 1, 2])]["SalePrice"].mean()
spring = data[data["Mo Sold"].isin([3, 4, 5])]["SalePrice"].mean()
season_diff = (spring - winter) / winter * 100
print(f"Цена весной выше цены зимой на {season_diff:.2f}%")
