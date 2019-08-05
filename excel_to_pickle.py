import pandas
import pickle
sheets = ["Answer Type Analysis Results", "Answer Length Analysis Results", "Not NE Answer Analysis Results"]
for sheet_name in sheets:
    sheet = pandas.read_excel("MRQA Results.xlsx", sheet_name=sheet_name)
    pickle.dump(sheet, open(sheet_name+".pkl", "wb"))
    #data = pickle.load(open(sheet_name+".pkl", "rb"))
