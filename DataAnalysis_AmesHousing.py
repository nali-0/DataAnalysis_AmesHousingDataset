import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def model_score(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_prediction = model.predict(X_te)
    r2 = r2_score(y_te, y_prediction)
    print(f"r2 = {r2:.4f}")

def main():
    data = pd.read_csv("AmesHousing.csv")
    data.drop(columns=["Order"], inplace=True)
    data.drop(columns=["PID"], inplace=True)

    for column in ["Alley", "Pool QC", "Fence", "Misc Feature", "Fireplace Qu", "Garage Type", "Garage Finish",
                   "Garage Qual", "Garage Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]:
        data[column] = data[column].fillna("None")

    for column in ["Bsmt Full Bath", "Bsmt Half Bath", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
                   "Total Bsmt SF", "Garage Area", "Garage Cars"]:
        data[column] = data[column].fillna(0)

    data["Lot Frontage"] = data.groupby("Neighborhood")["Lot Frontage"].transform(lambda x: x.fillna(x.median()))
    data["Lot Frontage"] = data["Lot Frontage"].fillna(data["Lot Frontage"].median())

    data["Mas Vnr Area"] = data["Mas Vnr Area"].fillna(0)
    data["Mas Vnr Type"] = data["Mas Vnr Type"].fillna("None")
    data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

    for col in data.columns[data.isnull().any()]:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)


    coef_data = pd.DataFrame({"feature": X.columns, "coef": ridge.coef_})
    coef_data["abs"] = np.abs(coef_data["coef"])
    top10 = coef_data.sort_values("abs", ascending=False).head(10)
    print("Ridge:")
    print(top10[["feature", "coef"]].to_string(index=False))


    slope, intercept, _, _, _ = stats.linregress(data["Gr Liv Area"], data["SalePrice"])
    data["residual"] = data["SalePrice"] - (slope * data["Gr Liv Area"] + intercept)

    z_scores = np.abs(stats.zscore(data["residual"].dropna()))

    anomalies = data[(z_scores > 3)]

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Gr Liv Area"], data["SalePrice"], alpha=0.5, label="All")
    plt.scatter(anomalies["Gr Liv Area"], anomalies["SalePrice"], color="red", label="Anomalies")
    plt.xlabel("Gr Liv Area")
    plt.ylabel("SalePrice")
    plt.legend()
    plt.show()

    model_score(X_scaled, y)

    anomaly_indices = anomalies.index
    X_cleared = np.delete(X_scaled, anomaly_indices, axis=0)
    y_cleared = y.drop(anomaly_indices)
    model_score(X_cleared, y_cleared)


    segments = ["Gr Liv Area", "Lot Area", "Overall Qual", "Overall Cond", "Year Built",
                "Total Bsmt SF", "Garage Area"]
    X_seg = data[segments].fillna(0)
    scaler_seg = StandardScaler()
    X_seg_scaled = scaler_seg.fit_transform(X_seg)

    kmeans = KMeans(n_clusters=5, n_init=10)
    data["Segment"] = kmeans.fit_predict(X_seg_scaled)
    print(data["Segment"].value_counts().sort_index())


    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pca, y, test_size=0.2)
    pca_model = LinearRegression()
    pca_model.fit(Xp_train, yp_train)
    yp_pred = pca_model.predict(Xp_test)
    print(f"PCA Regression: r2 = {r2_score(yp_test, yp_pred):.4f}")

    data["Age at Sale"] = data["Yr Sold"] - data["Year Built"]
    data["Years Since Remodel"] = data["Yr Sold"] - data["Year Remod/Add"]
    data["Age at Sale"] = data["Age at Sale"].clip(lower=0)
    data["Years Since Remodel"] = data["Years Since Remodel"].clip(lower=0)

    yearly = data.groupby("Yr Sold")["SalePrice"].mean()
    print("Average cost year:", yearly.round(0))

    monthly = data.groupby("Mo Sold")["SalePrice"].mean()
    print("Average cost month:", monthly.round(0))

    mean2007 = data[data["Yr Sold"] == 2007]["SalePrice"].mean()
    mean2009 = data[data["Yr Sold"] == 2009]["SalePrice"].mean()
    drop = (mean2009 - mean2007) / mean2007 * 100
    print(f"Drop 2007-2009 {drop:.1f}%")

    winter = data[data["Mo Sold"].isin([12, 1, 2])]["SalePrice"].mean()
    spring = data[data["Mo Sold"].isin([3, 4, 5])]["SalePrice"].mean()
    season_diff = (spring - winter) / winter * 100
    print(f"Spring vs Winter {season_diff:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(yearly.index, yearly.values, marker="o")
    axes[0].set_title("Year costs")
    axes[1].bar(monthly.index, monthly.values)
    axes[1].set_title("Monts Costs")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()