from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


# Data Preprocessing

train = (
    pd.read_csv("spaceship/data/train.csv")
    .drop(["Name"], axis=1)
    .set_index("PassengerId")
)
test_not_indexed = pd.read_csv("spaceship/data/test.csv").drop(["Name"], axis=1)
test = test_not_indexed.set_index("PassengerId")

train["Cabin0"] = train["Cabin"].str.split("/", expand=True)[0]
train["Cabin2"] = train["Cabin"].str.split("/", expand=True)[2]
y_train = train["Transported"]
train.drop(columns=["Cabin", "Transported"], inplace=True)
test["Cabin0"] = test["Cabin"].str.split("/", expand=True)[0]
test["Cabin2"] = test["Cabin"].str.split("/", expand=True)[2]
test.drop(columns=["Cabin"], inplace=True)

labeled_train = train.copy()
labeled_test = test.copy()

label_encoder_cols = ["HomePlanet", "Cabin0", "Cabin2", "Destination"]

encoder = OrdinalEncoder()
labeled_train[label_encoder_cols] = encoder.fit_transform(train[label_encoder_cols])
labeled_test[label_encoder_cols] = encoder.transform(test[label_encoder_cols])

imputer = SimpleImputer()

imputed_train = pd.DataFrame(imputer.fit_transform(labeled_train))
imputed_test = pd.DataFrame(imputer.transform(labeled_test))

imputed_train.columns = labeled_train.columns
imputed_train.index = labeled_train.index
imputed_test.columns = labeled_test.columns
imputed_test.index = labeled_test.index

# Model Prediction
model = DecisionTreeClassifier()

model.fit(imputed_train, y_train)
y_result = model.predict(imputed_test)

# Output result in kaggle format
y_result_pd = pd.DataFrame(y_result)
y_result_pd["PassengerId"] = test_not_indexed["PassengerId"]
y_result_pd.set_index("PassengerId", inplace=True)
y_result_pd.rename(columns={0: "Transported"}, inplace=True)
y_result_pd.to_csv("submission.csv")
