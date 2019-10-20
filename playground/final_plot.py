import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsfonts}']
matplotlib.rcParams['text.latex.preview'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 13
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

graphs_folder = Path('../graphs/')
fig, ax = plt.subplots(figsize=(11,4.4))
# Set the labels
ax.xaxis.set_label_text('Date')
ax.yaxis.set_label_text('Microsoft\'s Stock Price (\$)')
#ax.set_ylim(0.82, 0.9)
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
datas = ['../src/plot_data_baseline.csv', '../src/plot_data_baseline_sent.csv', '../src/plot_data_lstm.csv', '../src/plot_data_lstm_sent.csv']
i=0
for file in datas:
    dfs_base = pd.read_csv(file, index_col='date')
    converted_dates = list(map(datetime.datetime.strptime, dfs_base.index.values, len(dfs_base.index.values)*['%Y-%m-%d']))
    x_axis = converted_dates
    if i==0:
        line = ax.plot(x_axis, dfs_base['y_true'].values, linewidth=1, linestyle='-')[0]
        line.set_label('True value')

    line = ax.plot(x_axis, dfs_base['y_pred'].values, linewidth=1, linestyle='--')[0]
    if i==0:
        line.set_label('BP baseline')
    elif i==1:
        line.set_label('BP with sentiment volume')
    elif i==2:
        line.set_label('LSTM baseline')
    elif i==3:
        line.set_label('LSTM with sentiment volume')
    i +=1

formatter = mdates.DateFormatter('%d/%m/%Y')
ax.xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate(rotation=25)
#new_x = mdates.datestr2num(dfs.index.values)
# x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dfs.index.values]
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# line = ax.plot(new_x, dfs['y_pred'].values,
#                    linewidth=1)[0]
#plt.gcf().autofmt_xdate()
ax.legend()
fig.savefig(str(graphs_folder / 'finalGraph.eps'), bbox_inches='tight')
fig.show()
#fig.show()
print('tes')
