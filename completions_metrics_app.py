#interactive widget options ---> https://www.youtube.com/watch?v=_9WiB2PDO7k&ab_channel=JCharisTech%26J-Secur1ty

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

import cx_Oracle
con = cx_Oracle.connect('DATAMART_READ_ONLY/Welcome_1@p2date.eogresources.com')

import geopandas as gpd
import datetime
import plotly.graph_objects as go
import plotly as py 
import plotly.express as px

#-----
#FRAC 

#----------------------------------
# FRAC MAP

 #grab wells that are in the drilling, flowback, or completion phase

sSQL12 = ("""
select A.WELL_NAME, B.LATITUDE, B.LONGITUDE, A.PHASE_NAME, A.SUBPHASE_NAME, A.TEAM_NAME,
A.SCHED_FRAC_START_DATE,A.SCHED_SALES_DATE, A.FRAC_CREW_NAME, TVD,
A.FRAC_START_DATE, A.FRAC_END_DATE, A.PLAN_FLOWBACK_DATE, A.DO_PLUG_END_DATE, A.LAST_24HR_DESCR
from miso_dba.miso_well A, ODM_INFO.ODM_WELL B
where A.division_id = 63
AND A.PRIMO_PRPRTY = B.PRIMO_PRPRTY
AND A.SUBPHASE_NAME in ('FRACING', 'DO PLUGS')
AND EXTRACT(YEAR FROM A.SPUD_DATE)>2018

AND A.SPUD_DATE IS NOT NULL
AND a.PHASE_NAME= 'COMPLETION'
and b.latitude is not null
and a.last_active_date = trunc(sysdate)
order by b.longitude desc
""")

#grab data for drilling, flowback, and completion wells
df_completions = pd.read_sql(sSQL12, con=con) 
df_frac=df_completions[df_completions['SUBPHASE_NAME']=='FRACING']

#Just Frac Spreads for Map
df_fracspreads=df_frac.groupby('FRAC_CREW_NAME').first().reset_index()

#use geopandas to read the shape files, which are actually multiple files
map_df = gpd.read_file('C:\DailyReports\Completions_Morning\Texas_County_Boundaries_line')
map_area = gpd.read_file('C:\DailyReports\Completions_Morning\AREA_SHAPE_FILE\Area_Outlines.shp')

#create a blank matplotlib figure of 11.69 by 8.27
frac_map = plt.figure(figsize=(11.69, 8.27))
ax = plt.gca()

#plot first and second shapefiles
map_df.plot(ax = ax, color = 'grey')
map_area.plot(ax = ax, color = 'whitesmoke', edgecolor = 'k',alpha=1)

#zoom in on the map to just the area we are interested in
ax.set_xlim(xmin =-99.35, xmax = -97.0)
ax.set_ylim(ymin=28.1, ymax=29.8)

df_frac.plot.scatter(x='LONGITUDE',y='LATITUDE',color='g',
                     ax=ax, label = 'Wells being Fraced ({})'.format(len(df_frac)), 
                     s = 150, marker = 'v')

#this was labor intensive, but we are plotting the names of counties on the map
ax.set_facecolor('whitesmoke')
ax.text(-98.56,28.5,'McMullen')
ax.text(-98.24,28.5,'Live Oak')
ax.text(-97.85,28.5,'Bee')
ax.text(-99.13,28.57,'LaSalle')
ax.text(-98.65,29,'Atascosa')
ax.text(-99.025,28.9,'Frio')
ax.text(-98.15,29.2,'Wilson')
ax.text(-97.92,28.83,'Karnes')
ax.text(-97.44,29.05,'DeWitt')
ax.text(-97.6,29.45,'Gonzales')
ax.text(-97.5,28.65,'Goliad')
ax.text(-98.54,29.4,'Medina')

#plot the area numbers on the map
ax.text(-97.27,29.4,'7', color = 'purple', fontsize = 20)
ax.text(-97.45,29.32,'1', color = 'purple', fontsize = 20)
ax.text(-97.6,29.2,'8', color = 'purple', fontsize = 20)
ax.text(-97.81,29.05,'2', color = 'purple', fontsize = 20)
ax.text(-98.05,28.95,'3', color = 'purple', fontsize = 20)
ax.text(-98.375,28.875,'4', color = 'purple', fontsize = 20)
ax.text(-98.62,28.675,'5', color = 'purple', fontsize = 20)
ax.text(-99,28.425,'6', color = 'purple', fontsize = 20)

#title
ax.set_title('Frac Operations', fontsize = '25', color='black', style='oblique')
df_fracspreads = df_fracspreads.sort_values(by = 'LONGITUDE', ascending = True)

def placement1(xxx):
    temp=[]
    areas= [('AREA 6', 28.45), ('AREA 5', 28.5), ('AREA 4', 28.9), ('AREA 3',28.9), 
            ('AREA 2', 29.08), ('AREA 8', 29.2), ('AREA 1', 29.32), ('AREA 7', 29.4), ('UNKNOWN', 28.9)]
    total=len(df_fracspreads) #apply function?
    x=0
    zup=0
    zlow=0
    for index, row in df_fracspreads.iterrows():
        for area in areas:
            
            if x < len(df_fracspreads) and df_fracspreads['TEAM_NAME'].iloc[x] in area[0]:                
                if df_fracspreads['LATITUDE'].iloc[x] > area[1]:    # up side 
                        a= x*75 +50
                        b= x*35 +250
                        zup = zup +1
                        temp.append([x,a,b])
                        
                if df_fracspreads['LONGITUDE'].iloc[x] < area[1]:     # down side
                        a= x*70 +200
                        b= x*35 -0
                        zlow = zlow +1
                        temp.append([x,a,b])
                x = x+1      
        
    return temp  
temp = placement1(df_fracspreads)

for i, label2 in enumerate(df_fracspreads['FRAC_CREW_NAME']):
    
    x,a,b = temp[i]

    ax.annotate(label2+'\n', (df_fracspreads.LONGITUDE.iat[i],df_fracspreads.LATITUDE.iat[i]),
                ha='center', va='center',fontsize=13, bbox=dict(boxstyle='round,pad=0.2', fc='lightgreen', alpha=1),
                arrowprops=dict(arrowstyle='->', color = 'BLACK', linewidth = 1.2),
                textcoords='axes pixels',xytext=(a,b), color = 'black')
    
#for j, label2 in enumerate(df_plugs.WELL_NAME):
#    
#    ax.annotate(label2+'\n'+' TVD', (df_plugs.LONGITUDE.iat[i],df_plugs.LATITUDE.iat[i]),
#                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1),
#                arrowprops=dict(arrowstyle='->', color = 'blue', linewidth = 1.2),
#                textcoords='axes pixels',xytext=(0,-80), color = 'darkgreen' )
 
ax.set_yticklabels([]) # get rid of y axis labels
ax.set_xticklabels([])    # get rid of x axis labels

plt.legend(loc='lower right')



# TREATED LENGTH RECORDS

#---------------------------------------------------------------
sSQL2 = ("""
WITH X AS (

select STG.DIVISION_ID,
STG.PRIMO_PRPRTY,
WB.WELL_NAME,
STG.FRAC_COMPANY_ID,
CM.COMPANY_NAME AS FRAC_COMPANY,
STG.STAGE_NO,
STG.TRTM_START_DT,
STG.TRTM_END_DT,
COMP.PERF_INTERVAL,
CASE WHEN STG.TRTM_END_DT - TRUNC(STG.TRTM_END_DT) > (1/24*5) THEN TRUNC(STG.TRTM_END_DT)+1 ELSE TRUNC(STG.TRTM_END_DT) END AS CDC_DATE,
EXTRACT(YEAR FROM CASE WHEN STG.TRTM_END_DT - TRUNC(STG.TRTM_END_DT) > (1/24*5) THEN TRUNC(STG.TRTM_END_DT)+1 ELSE TRUNC(STG.TRTM_END_DT) END) AS CDC_YEAR
from ICOMP_DBA.CM_COMP_STAGE STG
LEFT JOIN ICOMP_DBA.IC_WELLBORE WB ON WB.PRIMO_PRPRTY=STG.PRIMO_PRPRTY
LEFT JOIN ICOMP_DBA.IC_COMPANY CM ON CM.COMPANY_ID=STG.FRAC_COMPANY_ID
LEFT JOIN ODM_INFO.ODM_COMP_STAGE COMP ON (COMP.PRIMO_PRPRTY=STG.PRIMO_PRPRTY) AND (COMP.PERC_ZONE_NBR=STG.STAGE_NO)
WHERE STG.DIVISION_ID IN (60,63)
AND STG.TRTM_END_DT >= TO_DATE('2017','YYYY')
)

SELECT 
X.FRAC_COMPANY,
COUNT(X.STAGE_NO) AS STAGE_COUNT,
SUM(X.PERF_INTERVAL) AS PERF_INTERVAL,
X.CDC_DATE,
X.CDC_YEAR
FROM X 
GROUP BY 
X.FRAC_COMPANY,
X.CDC_DATE,
X.CDC_YEAR
ORDER BY 
COUNT(X.STAGE_NO) DESC
--X.CDC_DATE
""")

df_test=pd.read_sql(sSQL2, con=con)
#df_test#['FRAC_COMPANY'].unique()

def df_zipper_companies():
    df_test['REAL_FRAC_CO']=df_test['FRAC_COMPANY']
    for i in range(0,df_test.index.max()):
        try:
            if "Evolution 4-A" in df_test['FRAC_COMPANY'].iloc[i]:
                df_test['REAL_FRAC_CO'].iloc[i]='EVOLUTION 4 SZ'
            elif "Evolution 4-B" in df_test['FRAC_COMPANY'].iloc[i]:
                df_test['REAL_FRAC_CO'].iloc[i]='EVOLUTION 4 SZ'
        except:
            pass
    return df_test

df=df_zipper_companies()
df_sz=df[df['REAL_FRAC_CO']=='EVOLUTION 4 SZ'].sort_values('CDC_DATE', ascending=False)
df_sz2=df_sz.groupby('CDC_DATE')['STAGE_COUNT','PERF_INTERVAL'].sum().reset_index()
df_sz2['REAL_FRAC_CO']='Evolution 4 SZ'
df_sz3=df_sz2.sort_values('STAGE_COUNT', ascending=False).reset_index().drop('index',1)

#display(df_sz3)
df_test2=df_test[['CDC_DATE','STAGE_COUNT','PERF_INTERVAL','REAL_FRAC_CO']]
df_stage_record=pd.concat([df_test2,df_sz3]).sort_values('STAGE_COUNT', ascending=False)
df_stage_record=df_stage_record[['REAL_FRAC_CO','STAGE_COUNT','PERF_INTERVAL','CDC_DATE']]
df_stage_record['FRAC_RECORDER']=df_stage_record['REAL_FRAC_CO'].astype(str)+'; '+df_stage_record['CDC_DATE'].astype(str)
df_length_record=df_stage_record.sort_values('PERF_INTERVAL',ascending=False).reset_index()

fig_length = px.bar(df_length_record.head(20), x='FRAC_RECORDER', y='PERF_INTERVAL',color='REAL_FRAC_CO',
color_discrete_map={"Evolution 4 SZ": "green",
 "Halliburton": "red",
 "Evolution 4": "green",
 "BJ Services1": "blue",
 "EVOLUTION 2":'lightgreen',
 'Evolution':'seagreen'},
 height=400, text='PERF_INTERVAL')
    
fig_length.update_layout(
    title="SA Treated Length Records",
    xaxis_tickangle=-45,
    xaxis_title="",
    yaxis_title="Treated Length in ft",
    margin=dict(l=80, r=80, t=100, b=90),
    xaxis = go.XAxis({'categoryorder':'total descending'},
        showticklabels=False)
)

#--------------------------------------------------------------

fig_stages = px.bar(df_stage_record.head(20), x='FRAC_RECORDER', y='STAGE_COUNT',color='REAL_FRAC_CO',
color_discrete_map={"Evolution 4 SZ": "green",
 "Halliburton": "red",
 "Evolution 4": "green",
 "BJ Services1": "blue",
 "EVOLUTION 2":'lightgreen',
 'Evolution':'seagreen'},
 height=400, text='STAGE_COUNT')
    
fig_stages.update_layout(
    title="SA Daily Stages Records",
    xaxis_tickangle=-45,
    xaxis_title="",
    yaxis_title="Stages #",
    margin=dict(l=80, r=80, t=100, b=90),
    xaxis = go.XAxis({'categoryorder':'total descending'},
        showticklabels=False)
)










sql1="""
select * from miso_dba.miso_well
where division_name= 'SAN ANTONIO'
and last_active_date = trunc(sysdate)
AND PHASE_NAME= 'COMPLETION'
AND EXTRACT(YEAR FROM SPUD_DATE)>2018
AND SUBPHASE_NAME in ('FRACING', 'DO PLUGS')
"""

def load_data():
    df = pd.read_sql(sql1, con=con).sort_values('FCST_IP_OIL')
    df['FCST_IP_OIL']=df['FCST_IP_OIL'].round(0).sort_values()
    return df

df=load_data()

fig_ip = px.bar(df, x='WELL_NAME', y='FCST_IP_OIL',
             color='TEAM_NAME',height=400, text='FCST_IP_OIL', hover_data=['FRAC_CREW_NAME','RORP', 'SUBTREND_NAME'])

fig_ip.update_layout(
    title="Forecast IP Oil",
    xaxis_tickangle=-45,
    xaxis_title="",
    yaxis_title="Forecast IP Oil (BOPD)",
    xaxis={'categoryorder':'total descending'},
    margin=dict(l=80, r=80, t=100, b=90)
)



#--------------------------------

#PLUGS - CT TUBING



sSQL6 = ("""
WITH X AS(
SELECT WELL.WELL_NAME, WELL.LONGITUDE, WELL.LATITUDE, WELL.TEAM_NAME,
A.*,
CASE WHEN A.TIME_TAGGED_PLUG_DT - TRUNC(A.TIME_TAGGED_PLUG_DT) > (1/24*5) THEN TRUNC(A.TIME_TAGGED_PLUG_DT)+1 ELSE TRUNC(A.TIME_TAGGED_PLUG_DT) END AS CDC_DATE 
FROM ICOMP_DBA.CM_COMP_PLUG A
LEFT JOIN ODM_INFO.ODM_WELL WELL ON A.PRIMO_PRPRTY=WELL.PRIMO_PRPRTY
WHERE COILED_TUBING_CO_CV_ID>0
ORDER BY COILED_TUBING_CO_CV_ID,
TIME_TAGGED_PLUG_DT
)

SELECT CDC_DATE, 
PRIMO_PRPRTY,
WELL_NAME,
TEAM_NAME,
LONGITUDE,
LATITUDE,
COILED_TUBING_CO_CV_ID AS CT_CREW,
COUNT(PLUG_NUMBER) AS TOTAL_PLUGS_PER_DAY
FROM X
WHERE CDC_DATE=TRUNC(SYSDATE)-1
GROUP BY COILED_TUBING_CO_CV_ID,
PRIMO_PRPRTY,
WELL_NAME,
TEAM_NAME,
LONGITUDE,
LATITUDE,
CDC_DATE
ORDER BY CDC_DATE DESC

""")

df_daily_plugs=pd.read_sql(sSQL6, con=con).sort_values('CT_CREW')
df_daily_plugs['CT_CREW']=df_daily_plugs['CT_CREW'].astype(str)
df_coil=df_daily_plugs[df_daily_plugs['LONGITUDE'].notna()]

#use geopandas to read the shape files, which are actually multiple files
map_df = gpd.read_file('C:\DailyReports\Completions_Morning\Texas_County_Boundaries_line')
map_area = gpd.read_file('C:\DailyReports\Completions_Morning\AREA_SHAPE_FILE\Area_Outlines.shp')

#create a blank matplotlib figure of 11.69 by 8.27
ct_map = plt.figure(figsize=(11.69, 8.27))
ax = plt.gca()

#plot first and second shapefiles
map_df.plot(ax = ax, color = 'grey')
map_area.plot(ax = ax, color = 'whitesmoke', edgecolor = 'k',alpha=1)

#zoom in on the map to just the area we are interested in
ax.set_xlim(xmin =-99.35, xmax = -97.0)
ax.set_ylim(ymin=28.1, ymax=29.8)

df_coil.plot.scatter(x='LONGITUDE',y='LATITUDE',color='orange',
                     ax=ax, label = 'Wells with CT ({})'.format(len(df_coil)), 
                     s = 150, marker = 'o')

#this was labor intensive, but we are plotting the names of counties on the map
ax.set_facecolor('whitesmoke')
ax.text(-98.56,28.5,'McMullen')
ax.text(-98.24,28.5,'Live Oak')
ax.text(-97.85,28.5,'Bee')
ax.text(-99.13,28.57,'LaSalle')
ax.text(-98.65,29,'Atascosa')
ax.text(-99.025,28.9,'Frio')
ax.text(-98.15,29.2,'Wilson')
ax.text(-97.92,28.83,'Karnes')
ax.text(-97.44,29.05,'DeWitt')
ax.text(-97.6,29.45,'Gonzales')
ax.text(-97.5,28.65,'Goliad')
ax.text(-98.54,29.4,'Medina')

#plot the area numbers on the map
ax.text(-97.27,29.4,'7', color = 'purple', fontsize = 20)
ax.text(-97.45,29.32,'1', color = 'purple', fontsize = 20)
ax.text(-97.6,29.2,'8', color = 'purple', fontsize = 20)
ax.text(-97.81,29.05,'2', color = 'purple', fontsize = 20)
ax.text(-98.05,28.95,'3', color = 'purple', fontsize = 20)
ax.text(-98.375,28.875,'4', color = 'purple', fontsize = 20)
ax.text(-98.62,28.675,'5', color = 'purple', fontsize = 20)
ax.text(-99,28.425,'6', color = 'purple', fontsize = 20)

#title
ax.set_title('Coil Tubing Operations', fontsize = '25', color='black', style='oblique')
df_coil = df_coil.sort_values(by = 'LONGITUDE', ascending = True)
 
ax.set_yticklabels([]) # get rid of y axis labels
ax.set_xticklabels([])    # get rid of x axis labels
plt.legend(loc='lower right')





#Daily Plugs
sSQL6 = ("""
WITH
    X
    AS
        (  SELECT A.*,
                  CASE
                      WHEN   A.TIME_TAGGED_PLUG_DT
                           - TRUNC (A.TIME_TAGGED_PLUG_DT) >
                           (1 / 24 * 5)
                      THEN
                          TRUNC (A.TIME_TAGGED_PLUG_DT) + 1
                      ELSE
                          TRUNC (A.TIME_TAGGED_PLUG_DT)
                  END    AS CDC_DATE
             FROM ICOMP_DBA.CM_COMP_PLUG A
            WHERE COILED_TUBING_CO_CV_ID > 0
         ORDER BY COILED_TUBING_CO_CV_ID, TIME_TAGGED_PLUG_DT)

  SELECT COILED_TUBING_CO_CV_ID     AS CT_CREW,
         CDC_DATE,
         COUNT (PLUG_NUMBER)        AS TOTAL_PLUGS_PER_DAY
    FROM X
   WHERE CDC_DATE = TRUNC (SYSDATE) - 1
GROUP BY COILED_TUBING_CO_CV_ID, CDC_DATE
ORDER BY CDC_DATE DESC
""")

df_daily_plugs=pd.read_sql(sSQL6, con=con)
df_daily_plugs['CT_CREW']=df_daily_plugs['CT_CREW'].astype(str)

fig_dailyplugs = px.bar(df_daily_plugs, x='CT_CREW', y='TOTAL_PLUGS_PER_DAY', color='CT_CREW',
color_discrete_map={"3922": "blue",
 "6004": "gold"},
 height=400, text='TOTAL_PLUGS_PER_DAY')
    
fig_dailyplugs.update_layout(
    title="SA Drilled Out Plugs YDY" + " "+str(df_daily_plugs['CDC_DATE'].iloc[0].strftime('%m/%d')),
    xaxis_tickangle=-45,
    xaxis_title="",
    yaxis_title="# of Plugs",
    margin=dict(l=80, r=80, t=100, b=90),
    xaxis = go.XAxis({'categoryorder':'total descending'},showticklabels=False)
)

# YDY Plugs Dataframe

sSQL6 = ("""
WITH X AS(
SELECT WELL.WELL_NAME,
A.*,
CASE WHEN A.TIME_TAGGED_PLUG_DT - TRUNC(A.TIME_TAGGED_PLUG_DT) > (1/24*5) THEN TRUNC(A.TIME_TAGGED_PLUG_DT)+1 ELSE TRUNC(A.TIME_TAGGED_PLUG_DT) END AS CDC_DATE 
FROM ICOMP_DBA.CM_COMP_PLUG A
LEFT JOIN ODM_INFO.ODM_WELL WELL ON A.PRIMO_PRPRTY=WELL.PRIMO_PRPRTY
WHERE COILED_TUBING_CO_CV_ID>0
ORDER BY COILED_TUBING_CO_CV_ID,
TIME_TAGGED_PLUG_DT
)

SELECT CDC_DATE, 
PRIMO_PRPRTY,
WELL_NAME,
COILED_TUBING_CO_CV_ID AS CT_CREW,
COUNT(PLUG_NUMBER) AS TOTAL_PLUGS_PER_DAY
FROM X
WHERE CDC_DATE=TRUNC(SYSDATE)-1
GROUP BY COILED_TUBING_CO_CV_ID,
PRIMO_PRPRTY,
WELL_NAME,
CDC_DATE
ORDER BY CDC_DATE DESC

""")

df_daily_plugs=pd.read_sql(sSQL6, con=con).sort_values('CT_CREW')
df_daily_plugs['CT_CREW']=df_daily_plugs['CT_CREW'].astype(str)

#Cumulative Plugs


sSQL1 = ("""
WITH X AS(
SELECT 
A.*,
CASE WHEN A.TIME_TAGGED_PLUG_DT - TRUNC(A.TIME_TAGGED_PLUG_DT) > (1/24*5) THEN TRUNC(A.TIME_TAGGED_PLUG_DT)+1 ELSE TRUNC(A.TIME_TAGGED_PLUG_DT) END AS CDC_DATE 
FROM ICOMP_DBA.CM_COMP_PLUG A
WHERE COILED_TUBING_CO_CV_ID>0
ORDER BY COILED_TUBING_CO_CV_ID,
TIME_TAGGED_PLUG_DT
)

SELECT CDC_DATE,
COILED_TUBING_CO_CV_ID AS CREW,
COUNT(PLUG_NUMBER) AS PLUGS
FROM X
WHERE to_char(sysdate, 'yyyy' ) = to_char(X.CDC_DATE, 'yyyy' )
GROUP BY COILED_TUBING_CO_CV_ID,
CDC_DATE
ORDER BY CDC_DATE
""")

df=pd.read_sql(sSQL1, con=con)

df['PLUGS_CUM']=df.groupby('CREW')['PLUGS'].cumsum()

fig_cumplugs = px.line(df, x="CDC_DATE", y="PLUGS_CUM", color='CREW')

fig_cumplugs.update_layout(
    title="Cumulative Plugs This Year",
    xaxis_title="Date",
    yaxis_title="# of Plugs",
    margin=dict(l=80, r=80, t=100, b=90)
)






#def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
st.sidebar.title("Completions Metrics")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Frac", "Coil Tubing","IP Oil Forecast"])
if app_mode == "Frac":
    st.write("""
    # Frac Daily Metrics
    Shown are metrics such as **footage/day** and ***stages/day***
    """)
    st.pyplot(frac_map)
    st.write("""
    # Frac Records 
    """)
    st.plotly_chart(fig_stages)
    st.plotly_chart(fig_length)
    
elif app_mode == "Coil Tubing":
    st.write("""
    # Coil Tubing Daily Metrics
    """)
    st.pyplot(ct_map)
    st.plotly_chart(fig_dailyplugs)
    df_daily_plugs
    st.plotly_chart(fig_cumplugs)

elif app_mode == "IP Oil Forecast":
    st.write("""
    # IP Oil Forecasts 
    """)
    st.plotly_chart(fig_ip)


















