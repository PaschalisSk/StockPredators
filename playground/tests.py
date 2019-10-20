import pandas as pd

micr_results = pd.read_json('../SemEval2017Task5/microsoft.sent.json')
micr_results.sort_values('sentiment score', inplace=True)
micr_results2 = pd.read_json('../SemEval2017Task5/submission.json')
micr_results2.sort_values('sentiment score', inplace=True)
print('test')
