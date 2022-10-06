import psycopg2
import pandas.io.sql as sqlio
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


#Database connection
with psycopg2.connect(
   database="Endeavor_SQL_tasks", user='Tom', password=os.environ['endeavor_pass'], host='127.0.0.1', port= '5432') as conn:  
   # Get data from views
   sql = "SELECT * FROM weekly_totals;"
   dat = sqlio.read_sql_query(sql, conn)
   sql = "SELECT date, week_id FROM joined_tbl;"
   dates = sqlio.read_sql_query(sql, conn, parse_dates=['date'])

def merge_dates(data, dates):
   """merges the mean date onto the data and number of days in week_id
   to be used in the plot.s

   Args:
       data (pd.Dataframe): weekly totals data~
       dates (pd.Dataframe): all dates in data and week_ids
   """
   dates_info = dates.groupby('week_id').agg({'date': ['mean', 'count']})
   return data.merge(dates_info, how='left', left_on='week_id', right_index=True)
   
dat = merge_dates(dat, dates)

fig, ax1 = plt.subplots(1,1, figsize=(10,5) )

ax1.bar(dat[('date', 'mean')], dat['total_revenue']/10**3, color='#93C47D', label = 'Revenue', width=dat[('date', 'count')]-0.6)
ax1.bar(dat[('date', 'mean')], 0-dat['total_spend']/10**3, color='#E06666', label = 'Costs', width=dat[('date', 'count')]-0.6)
ax1.plot(dat[('date', 'mean')], (dat['total_revenue']-dat['total_spend'])/10**3, color='k', label='Profit')


ax1.spines['top'].set_color('none')
ax1.xaxis.tick_bottom()
ax1.spines['bottom'].set_position('zero')
ax1.spines['right'].set_color('none')
ax1.tick_params(axis='x', pad=50)
ax1.yaxis.set_major_formatter('£{x:1.0f}K')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.yaxis.tick_left()
ax1.legend()


plt.tight_layout()
plt.show()
plt.close()

