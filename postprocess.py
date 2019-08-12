import pandas
import sys


path = sys.argv[1]
df = pandas.read_csv(path, sep="\t")

gold_type = df["gold-type"].tolist()
em = df["em"].tolist()
counter = {}

for i in range(0, len(df)):
    counter.setdefault(gold_type[i], {})
    counter[gold_type[i]]["total"] = counter[gold_type[i]].get("total", 0) + 1
    counter[gold_type[i]][em[i]] = counter[gold_type[i]].get(em[i], 0) + 1

print(counter)
column = ["type", "total", "Percentage", "EM==True", "Ratio"]
values = []

for k, v in counter.items():
    true_number = v.get(True, 0)
    values.append([k, v["total"], "{:.2f}%".format(v["total"]/len(em)*100), true_number, \
                   "{:.2f}%".format(true_number/v["total"]*100)])

new_df = pandas.DataFrame(values)
new_df.columns = column
new_df.to_csv(sys.argv[2])

