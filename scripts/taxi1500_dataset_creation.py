import json

import pandas as pd

data_path = "data/Taxi1500/raw"
# load ground truth
eng_train = pd.read_table(
    f"{data_path}/eng_data/eng_train.tsv",
    delimiter="\t",
    header=None,
    names=["id", "classification", "text"],
).to_numpy()

eng_test = pd.read_table(
    f"{data_path}/eng_data/eng_test.tsv",
    delimiter="\t",
    header=None,
    names=["id", "classification", "text"],
).to_numpy()

eng_val = pd.read_table(
    f"{data_path}/eng_data/eng_dev.tsv",
    delimiter="\t",
    header=None,
    names=["id", "classification", "text"],
).to_numpy()

# load different languages
fr_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/fra-x-bible-perret.txt",
    sep="\t",
    header=None,
    skiprows=11,
    names=["id", "text"],
).to_numpy()

de_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/deu_deu-f35.ebible.txt",
    sep="\t",
    header=None,
    skiprows=10,
    names=["id", "text"],
).to_numpy()

esp_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/spa_spavbl.ebible.txt",
    sep="\t",
    header=None,
    skiprows=22,
    names=["id", "text"],
).to_numpy()

ita_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/ita-x-bible-2009.txt",
    sep="\t",
    header=None,
    skiprows=12,
    names=["id", "text"],
).to_numpy()

por_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/por_porbrbsl.ebible.txt",
    sep="\t",
    header=None,
    skiprows=14,
    names=["id", "text"],
).to_numpy()

arb_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/arb-x-bible.txt",
    sep="\t",
    header=None,
    skiprows=12,
    names=["id", "text"],
).to_numpy()

grk_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/grc_grc-tisch.ebible.txt",
    sep="\t",
    header=None,
    skiprows=22,
    names=["id", "text"],
).to_numpy()

rus_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/rus_russyn.ebible.txt",
    sep="\t",
    header=None,
    skiprows=8,
    names=["id", "text"],
).to_numpy()

nab_data = pd.read_csv(
    f"{data_path}/Taxi1500-c_v3.0/naf_naf.ebible.txt",
    sep="\t",
    header=None,
    skiprows=20,
    names=["id", "text"],
).to_numpy()


train_ds = []

for i in range(len(eng_train)):
    # get metadata
    row_dict = {}
    row_dict["id"] = eng_train[i][0]
    row_dict["class"] = eng_train[i][1]
    # get translations
    languages = {}
    languages["english"] = eng_train[i][2]
    languages["french"] = fr_data[fr_data[:, 0] == row_dict["id"]][0][1]
    languages["german"] = de_data[de_data[:, 0] == row_dict["id"]][0][1]
    languages["spanish"] = esp_data[esp_data[:, 0] == row_dict["id"]][0][1]
    languages["portuguese"] = por_data[por_data[:, 0] == row_dict["id"]][0][1]
    languages["italian"] = ita_data[ita_data[:, 0] == row_dict["id"]][0][1]
    languages["greek"] = grk_data[grk_data[:, 0] == row_dict["id"]][0][1]
    languages["russian"] = rus_data[rus_data[:, 0] == row_dict["id"]][0][1]
    languages["arabic"] = arb_data[arb_data[:, 0] == row_dict["id"]][0][1]
    languages["nabak"] = nab_data[nab_data[:, 0] == row_dict["id"]][0][1]
    row_dict["languages"] = languages
    # add to dict
    train_ds.append(row_dict)

with open("data/Taxi1500/processed/train_data.json", "w") as f:
    json.dump(train_ds, f, indent=2)

test_ds = []

for i in range(len(eng_test)):
    # get metadata
    row_dict = {}
    row_dict["id"] = eng_test[i][0]
    row_dict["class"] = eng_test[i][1]
    # get translations
    languages = {}
    languages["english"] = eng_test[i][2]
    languages["french"] = fr_data[fr_data[:, 0] == row_dict["id"]][0][1]
    languages["german"] = de_data[de_data[:, 0] == row_dict["id"]][0][1]
    languages["spanish"] = esp_data[esp_data[:, 0] == row_dict["id"]][0][1]
    languages["portuguese"] = por_data[por_data[:, 0] == row_dict["id"]][0][1]
    languages["italian"] = ita_data[ita_data[:, 0] == row_dict["id"]][0][1]
    languages["greek"] = grk_data[grk_data[:, 0] == row_dict["id"]][0][1]
    languages["russian"] = rus_data[rus_data[:, 0] == row_dict["id"]][0][1]
    languages["arabic"] = arb_data[arb_data[:, 0] == row_dict["id"]][0][1]
    languages["nabak"] = nab_data[nab_data[:, 0] == row_dict["id"]][0][1]
    row_dict["languages"] = languages
    # add to dict
    test_ds.append(row_dict)

with open("data/Taxi1500/processed/test_data.json", "w") as f:
    json.dump(test_ds, f, indent=2)

val_ds = []

for i in range(len(eng_val)):
    # get metadata
    row_dict = {}
    row_dict["id"] = eng_val[i][0]
    row_dict["class"] = eng_val[i][1]
    # get translations
    languages = {}
    languages["english"] = eng_val[i][2]
    languages["french"] = fr_data[fr_data[:, 0] == row_dict["id"]][0][1]
    languages["german"] = de_data[de_data[:, 0] == row_dict["id"]][0][1]
    languages["spanish"] = esp_data[esp_data[:, 0] == row_dict["id"]][0][1]
    languages["portuguese"] = por_data[por_data[:, 0] == row_dict["id"]][0][1]
    languages["italian"] = ita_data[ita_data[:, 0] == row_dict["id"]][0][1]
    languages["greek"] = grk_data[grk_data[:, 0] == row_dict["id"]][0][1]
    languages["russian"] = rus_data[rus_data[:, 0] == row_dict["id"]][0][1]
    languages["arabic"] = arb_data[arb_data[:, 0] == row_dict["id"]][0][1]
    languages["nabak"] = nab_data[nab_data[:, 0] == row_dict["id"]][0][1]
    row_dict["languages"] = languages
    # add to dict
    val_ds.append(row_dict)

with open("data/Taxi1500/processed/val_data.json", "w") as f:
    json.dump(val_ds, f, indent=2)

print(json.dumps(train_ds[0], indent=3))
