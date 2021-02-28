import os
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from scipy.spatial import distance
from scipy.spatial import distance_matrix
pd.set_option('display.max_columns', None)

#https://www.youtube.com/watch?v=ZZ4B0QUHuNc&ab_channel=DataProfessor


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

st.write("""
# AUT Inspection App
Demo Data Web app, project idea by ***Nick Malinaric***!
""")

st.write("""
## Inspector Name
""")
st.text_input('Enter Inspector Name')

st.write("""
## Inspection Date
""")
st.date_input('Enter Inspection Date')

st.write("""
## Vessel ID
""")
st.number_input('Enter Vessel ID', min_value=0, step=1, value=10000)

st.write("""
## Manufacturer Serial Number
""")
st.number_input('Enter Manufacturing SN', min_value=0, step=1, value=141000000)

st.write("""
## Vessel Operating Start Date
""")
st.date_input('Enter Veseel Start Date')

st.write("""
## Excel File Upload
""")
excel_file=st.file_uploader('Upload excel file')


#-----------------------------------------------

#Importing Data from excel sheet and replacing all 0 with numpy nans
df=pd.read_excel(excel_file)
df=df.set_index(0.00)
df=df.replace(0, np.nan)

#Convert grid data to x, y, z
df_xyz=df.reset_index().set_index(0.00).stack().reset_index(name='z').rename(columns={'level_1':'x', 0.0:'y'})
df_xyz['TARGET']='none'
df_xyz['TARGET_NUMBER']=0
df_backup=df_xyz
df_xyz['x']=df_xyz['x'].astype(float)

#Copy of df_xyz for later use
stack_df=pd.DataFrame(df.stack())
stack_df=pd.DataFrame(stack_df.to_records()).rename(columns={'0.0':'y', 'level_0':'x', '0':'z'}).sort_values(['x','y'], ascending=True)
stack_df['x']=stack_df['x'].astype(float)

#Loss per cell
df_wall_loss=1.5-df

#number of scans
total_number_of_scans=int(len(df_wall_loss.stack()))

#smallest thickness of grid
absolute_minimum_reading=df_xyz['z'].min()

#average_wall_thickness
average_wall_thickness=stack_df['z'].mean()

#average_wall_loss
average_wall_loss=df_wall_loss.stack().mean() 
std_wall_loss=df_wall_loss.stack().std()  

#First Summary Graph of readings
dd1 = {'Total # of Readings': [total_number_of_scans],
      'Avg Wall Thickness': [average_wall_thickness],
      'Min Measured Thickness': [absolute_minimum_reading]
     }
ddf1 = pd.DataFrame(data=dd1)
ddf1=ddf1.style.hide_index()




# Grid Blocks Work
# Grid block dataframe with step change
def block_grid(step_change):
    d = {'i':[],'j':[],'block_avg':[]}
    for i in range(0,len(df.columns)+1 , step_change):
        for j in range(0,len(df.index)+1 , step_change):
            df_block=df[df.columns.values[i:i+step_change]].iloc[j:j+step_change]
            block_avg=df_block.values.mean()
            d['i'].append(i)
            d['j'].append(j)
            d['block_avg'].append(block_avg)

    df_block= pd.DataFrame(d)
    df_nick=df_block.pivot(index="j", columns="i", values="block_avg")
    return df_nick

#Pretty Map showing whole Separator Surface, takes a while to load
def block_grid_heatmap(step_change):
    fig, ax = plt.subplots()
    sns.heatmap(block_grid(step_change),vmin=0.5, vmax=1.5, cmap=ListedColormap(['red', 'orange', 'yellow', 'lightgreen',  'green']))
    ax.tick_params(left=False, bottom=False)
    plt.title('Thickness Map for '+str(step_change)+' X '+str(step_change)+' grid size')
    return plt.show()

#Needed Dataframe for CFD graph of wall thicknesses 
def prob_block(step_change):
    df_try=block_grid(2)
    df_try2=df_try.reset_index()#.values.tolist()
    df_try3=df_try2.loc[:, df_try2.columns != 'j'] #remove j out of dataframe
    value_list=df_try3.values.flatten().tolist() # flatten() to prevent 2d array
    df_try4 = pd.DataFrame(value_list,columns=['average_value'])
    df_try5= df_try4.dropna().sort_values('average_value', ascending=True).reset_index().drop('index', axis=1)
    df_try5['CFD']=df_try5.index/(len(df_try5)-1)
    return df_try5

#Actual Graph
def prob_graph(step_change):
    sns.lineplot(prob_block(step_change)['average_value'], prob_block(step_change)['CFD'], color='orange')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.xlabel('Wall Thickness (in)')
    return plt.show()



#Parameters and General Calculations
design_cond_pressure=500 #psig P
design_cond_temp=450 #Fahrenheit
inside_diameter=48#inches
wall_thickness=1.5 #inches
uniform_metal_loss=average_wall_loss #inches
future_corrosion_allowance=0.1 #wall_thickness-average_wall_loss #inches
material= 'SA 516 Grade 70'
weld_joint_efficiency=0.7 #unitless E
allowable_tensile_stress=17500

RSFa=0.9

nominal_wall_thickness=1.5 #original wall thickness after fabrication
t=wall_thickness-average_wall_loss #same as average_wall_thickness

#Local Metal Loss Functions
def radius_circ():
    return (inside_diameter/2)+uniform_metal_loss+future_corrosion_allowance
#radius_circ()

def minimum_required_thickness_circ():
    return (design_cond_pressure*radius_circ())/((allowable_tensile_stress*weld_joint_efficiency)-0.6*design_cond_pressure)

def minimum_required_thickness_long():
    return (design_cond_pressure*radius_circ())/((2*allowable_tensile_stress*weld_joint_efficiency)+0.4*design_cond_pressure)

def minimum_required_thickness():
    return max(minimum_required_thickness_circ(), minimum_required_thickness_long())

#Second Summary Table for Calculated Values of Minimum Required Thickness
dd2 = {'Min Req Thickness (circ)': [minimum_required_thickness_circ()],
      'Min Req Thickness (long)': [minimum_required_thickness_long()],
      'Min Req Thickness': [minimum_required_thickness()]
     }
ddf2 = pd.DataFrame(data=dd2)
ddf2=ddf2.style.hide_index()





#Wall Loss Severity Ranking
for i in range(df_xyz.index.min(), df_xyz.index.max()+1):
    if df_xyz['z'].iloc[i]<0.75:
        df_xyz['TARGET'].iloc[i]='EXTREME'
        df_xyz['TARGET_NUMBER'].iloc[i]=5
    elif (df_xyz['z'].iloc[i]>=0.75) and (df_xyz['z'].iloc[i]<1.0):
        df_xyz['TARGET'].iloc[i]='CRITICAL_LOSS'
        df_xyz['TARGET_NUMBER'].iloc[i]=4
    elif (df_xyz['z'].iloc[i]>=1.0) and (df_xyz['z'].iloc[i]<1.25):
        df_xyz['TARGET'].iloc[i]='SUBSTANTIAL_LOSS'
        df_xyz['TARGET_NUMBER'].iloc[i]=3
    elif (df_xyz['z'].iloc[i]>=1.25) and (df_xyz['z'].iloc[i]<1.5):
        df_xyz['TARGET'].iloc[i]='WALL_LOSS'
        df_xyz['TARGET_NUMBER'].iloc[i]=2
    elif (df_xyz['z'].iloc[i]>=1.5):
        df_xyz['TARGET'].iloc[i]='PRISTINE'   
        df_xyz['TARGET_NUMBER'].iloc[i]=1
df_xyz=df_xyz.sort_values(['x','y'], ascending=[True, True])


# Looking at Risk 5 Only
df_red=df_xyz[df_xyz['TARGET_NUMBER']>=4][['x','y','z','TARGET','TARGET_NUMBER']].sort_values(['x','y'], ascending=[True,True]).reset_index().drop('index',1)
df_red['delta_x']=df_red['x']-df_red['x'].shift(1)
df_red['delta_y']=df_red['y']-df_red['y'].shift(1)
df_red[['delta_x','delta_y']]=df_red[['delta_x','delta_y']].fillna(0)
df_red['cluster']='none'

# Clusterization of Extreme Points
index_range=range(df_red.index.min(), df_red.index.max()+1)
index_list = list(index_range)
i=1 #First row of new group 
while i <= 1:
    for a in index_list:
        df_red['cluster'].iloc[a-1]=i
        if (df_red['delta_x'].iloc[a]<=0.25)&(abs(df_red['delta_y'].iloc[a])<=0.25):
            df_red['cluster'].iloc[a]=i
        else:
            i+=1

#Because my loops above won't give a correct value for the last row, code below            
if (df_red['delta_x'].iloc[index_list[-1]]<=0.25)&(abs(df_red['delta_y'].iloc[index_list[-1]])<=0.5):
    df_red['cluster'].iloc[index_list[-1]]=i
else:
    df_red['cluster'].iloc[index_list[-1]]=df_red['cluster'].max()+1

df_red['cluster']=df_red['cluster'].astype(int)
            
df_sum=df_red

# Bulk of Cluster Parameter Summary
#If cluster is smaller than 7 in2 than it can be considered as a pit, anything bigger than 7 in2 is either local metal loss or general metal loss
df_x_area=pd.DataFrame(df_sum.groupby('cluster').agg({'x':['min','max']}).reset_index().to_records())
df_y_area=pd.DataFrame(df_sum.groupby('cluster').agg({'y':['min','max']}).reset_index().to_records())
df_cluster_size=df_sum.groupby('cluster').size().reset_index().rename(columns={0:'size'})
df_surface_area1=df_x_area.merge(df_y_area, left_index=True, right_index=True)
df_surface_area2=df_surface_area1.merge(df_cluster_size, left_index=True, right_index=True)
df_cluster_area=df_surface_area2.drop(['index_x','index_y',"('cluster', '')_y","cluster"], axis=1).rename(columns={"('cluster', '')_x":"cluster", "('x', 'min')":"x_min_cluster", "('x', 'max')":"x_max_cluster", 
"('y', 'min')": "y_min_cluster", "('y', 'max')":"y_max_cluster"})
#df_cluster_area["surface_area"]=df_cluster_area['']
df_cluster_area['xcenter']=(df_cluster_area['x_max_cluster']+df_cluster_area['x_min_cluster'])/2
df_cluster_area['ycenter']=(df_cluster_area['y_max_cluster']+df_cluster_area['y_min_cluster'])/2
df_cluster_area['area']=df_cluster_area['size']*0.25*0.25
df_cluster_area['s']=df_cluster_area['x_max_cluster']-df_cluster_area['x_min_cluster']
df_cluster_area['c']=df_cluster_area['y_max_cluster']-df_cluster_area['y_min_cluster']
df_cluster_area[['s','c']]=df_cluster_area[['s','c']]+0.25

#List of All Clusters
cluster_list=df_cluster_area['cluster'].tolist()
number_of_pits=len(cluster_list)

#Get Area for Each Cluster
def cluster_data(cluster_number):
    return df_sum[df_sum['cluster']==cluster_number]

#Using cluster_data function, get specific info for each cluster
def cluster_xmin(cluster_number):
    return cluster_data(cluster_number)['x'].min()
#cluster_xmin(1)

def cluster_xmax(cluster_number):
    return cluster_data(cluster_number)['x'].max()
#cluster_xmax(1)

def cluster_ymin(cluster_number):
    return cluster_data(cluster_number)['y'].min()
#cluster_ymin(1)

def cluster_ymax(cluster_number):
    return cluster_data(cluster_number)['y'].max()
#cluster_ymax(1)

def s_cluster(cluster_number):
    return cluster_data(cluster_number)['x'].max()-cluster_data(cluster_number)['x'].min()
#s_cluster(1)

def c_cluster(cluster_number):
    return cluster_data(cluster_number)['y'].max()-cluster_data(cluster_number)['y'].min()

def xcenter_cluster(cluster_number):
    return (cluster_xmin(cluster_number)+cluster_xmax(cluster_number))/2
#xcenter_cluster(1)

def ycenter_cluster(cluster_number):
    return (cluster_ymin(cluster_number)+cluster_ymax(cluster_number))/2

d = {'cluster':[],'xcenter':[], 'ycenter':[],'xmin':[], 'xmax':[],'ymin':[],'ymax':[],'s':[], 'c':[]}
for cluster in cluster_list:
    d['cluster'].append(cluster)
    d['xcenter'].append(xcenter_cluster(cluster))
    d['ycenter'].append(ycenter_cluster(cluster))
    d['xmin'].append(cluster_xmin(cluster))
    d['xmax'].append(cluster_xmax(cluster))
    d['ymin'].append(cluster_ymin(cluster))
    d['ymax'].append(cluster_ymax(cluster))
    d['s'].append(s_cluster(cluster))
    d['c'].append(c_cluster(cluster))
df_cluster_info=pd.DataFrame(d)



# Better Alternative for df_cluster_info
df_work=df_cluster_area.sort_values('area', ascending=False)
df_work['problem_class']='none'
for i in range(df_work.index.min(), df_work.index.max()+1):
    if df_work['area'].iloc[i]>=1:
        df_work['problem_class'].iloc[i]='LOCAL_LOSS' 
    elif (df_work['area'].iloc[i]<=1) and (df_work['area'].iloc[i]>0.0625):
        df_work['problem_class'].iloc[i]='PITTING' 
    elif df_work['size'].iloc[i]==1:
        df_work['problem_class'].iloc[i]='OUTLIER'  
df_work=df_work.reset_index().drop('index', axis=1)




# Local Loss
df_local_loss=df_work[df_work['problem_class']=='LOCAL_LOSS']

# Specific Local Loss Cluster Data
def local_loss_cn(cluster_number):
     return df_local_loss[df_local_loss['cluster']==cluster_number]

# Specific minimum measured thickness for each cluster
def minimum_measured_thickness_local(cluster_number):
    return cluster_data(cluster_number)['z'].min()

# Specific remaining_thickness_ratio for each cluster
def remaining_thickness_ratio_local(cluster_number):
    rtr_local=(minimum_measured_thickness_local(cluster_number)-future_corrosion_allowance)/minimum_required_thickness()
    if rtr_local<0:
        rtr_local2=0
    else: 
        rtr_local2=rtr_local
    return rtr_local2

def s_local_loss(cluster_number):
    return local_loss_cn(cluster_number)['s'].iloc[0]
#s_local_loss(48)

def c_local_loss(cluster_number):
    return local_loss_cn(cluster_number)['c'].iloc[0]
#c_local_loss(48)

def shell_parameter_local(cluster_number):
    return (1.285*s_local_loss(cluster_number))/((inside_diameter*minimum_required_thickness())**0.5)
#shell_parameter_local(48)

def Mt_local(cluster_number):
    return (1+0.48*shell_parameter_local(cluster_number)**2)**0.5
#Mt_local(48)

def RSF_local(cluster_number):
    return remaining_thickness_ratio_local(cluster_number)/(1-(1/Mt_local(cluster_number))*(1-remaining_thickness_ratio_local(cluster_number)))

def c_over_d(cluster_number):
    return c_local_loss(cluster_number)/inside_diameter
#c_over_d(48)

def cond1_local_loss(cluster_number):
    if (c_over_d(cluster_number)<=0.348)&(remaining_thickness_ratio_local(cluster_number)>=0.2):
        print('ACCEPTABLE')   
    elif (c_over_d(cluster_number)<=0.348)&(remaining_thickness_ratio_local(cluster_number)<0.2):
        print('UNACCEPTABLE')

# Includes Local Loss Criteria and Conditions for Safe Use
df_work2=df_work
df_work2['t_mm']='none'
df_work2['R_t']='none'
df_work2['lambda']='none'
df_work2['M_t']='none'
df_work2['RSF']='none'
df_work2['c/d']='none'
for i in range(df_work2.index.min(), df_work2.index.max()+1):
    if df_work2['problem_class'].iloc[i]=='LOCAL_LOSS':
        df_work2['t_mm'].iloc[i]=minimum_measured_thickness_local(df_work2['cluster'].iloc[i])
        df_work2['R_t'].iloc[i]=remaining_thickness_ratio_local(df_work2['cluster'].iloc[i])
        df_work2['lambda'].iloc[i]=shell_parameter_local(df_work2['cluster'].iloc[i])
        df_work2['M_t'].iloc[i]=Mt_local(df_work2['cluster'].iloc[i])
        df_work2['RSF'].iloc[i]=RSF_local(df_work2['cluster'].iloc[i])
        df_work2['c/d'].iloc[i]=c_over_d(df_work2['cluster'].iloc[i])




# Pit Couple Identification
df_pitting=df_work2[df_work2['problem_class']=='PITTING'].reset_index().drop('index',axis=1)

#Distance Matrix for each cluster to every other cluster, distance
coord=df_pitting[['xcenter','ycenter']].to_numpy()
df_centers=df_pitting[['cluster','xcenter','ycenter']]
df_centers=df_centers.set_index('cluster')
df_matrix=pd.DataFrame(distance_matrix(df_centers.values, df_centers.values), index=df_centers.index, columns=df_centers.index)

#Convert Distance Matrix into Dataframe
df_stack=pd.DataFrame(df_matrix.stack())
df_stack=df_stack.rename(columns={0:'distance'})
df_stack=df_stack.reset_index(level=0)
df_stack=df_stack.rename(columns={'cluster':'cluster2'})
df_exe=pd.DataFrame(df_stack.to_records())
df_all_pit_couples=df_exe[df_exe['distance']>0].sort_values('distance').reset_index()

# Daniel
def df_new_pit_couple(df):
    return df.head(1)
#df_new_pit_couple(df_all_pit_couples)

def new_pit_couple_list(df):
    return df_new_pit_couple(df)[['cluster','cluster2']].iloc[0].tolist()
#new_pit_couple_list(df_all_pit_couples)

def df_rem_pit_couples(df):
    return df[(~df['cluster'].isin(new_pit_couple_list(df)))&(~df['cluster2'].isin(new_pit_couple_list(df)))]
#df_rem_pit_couples(df_rem_pit_couples(df_all_pit_couples))

def find_all_min_clusters_new(df):
    cluster_list = []
    cluster_list_2 = []
    distance_list = []
    for row in df.itertuples():
        if (row[2] not in cluster_list) and (row[3] not in cluster_list) and (row[2] not in cluster_list_2) and (row[3] not in cluster_list_2):
            cluster_list.append(row[2])
            cluster_list_2.append(row[3])
            distance_list.append(row[4])
        else:
            pass
    return pd.DataFrame({'first_cluster':cluster_list,'second_cluster':cluster_list_2, 'distance':distance_list})

# Duplicate Code with cluster_data()?
def df_local_zone(cluster_number):
    df=df_cluster_info[df_cluster_info['cluster']==cluster_number]
    df_local_zone= stack_df[(stack_df['x']>=df['xmin'].iloc[0])&(stack_df['x']<=df['xmax'].iloc[0])&(stack_df['y']>=df['ymin'].iloc[0])&(stack_df['y']<=df['ymax'].iloc[0])]
    return df_local_zone[['x','y','z']]
#df_local_zone(0)

# Duplicate Code
def min_thickness(cluster_number):
    return df_local_zone(cluster_number)['z'].min()

def w_pit(cluster_number):
    return nominal_wall_thickness-min_thickness(cluster_number)

# Calculate necessary parameters for criteria later
df_pc=find_all_min_clusters_new(df_all_pit_couples)
df_pc['wi']='none'
df_pc['wj']='none'
df_pc['di']='none'
df_pc['dj']='none'
for i in range(df_pc.index.min(), df_pc.index.max()+1):
    df_pc['wi'].iloc[i]=w_pit(df_pc['first_cluster'].iloc[i])
    df_pc['wj'].iloc[i]=w_pit(df_pc['second_cluster'].iloc[i])
    
    df_try1=df_cluster_info[df_cluster_info['cluster']==df_pc['first_cluster'].iloc[i]]   
    df_pc['di'].iloc[i]=max(df_try1['s'].iloc[0],df_try1['c'].iloc[0] )
    
    df_try2=df_cluster_info[df_cluster_info['cluster']==df_pc['second_cluster'].iloc[i]]   
    df_pc['dj'].iloc[i]=max(df_try2['s'].iloc[0],df_try2['c'].iloc[0] )




# Widespread Pitting
df_pc['actual_depth_i']=df_pc['wi']-(t-future_corrosion_allowance-minimum_required_thickness())
df_pc['actual_depth_j']=df_pc['wj']-(t-future_corrosion_allowance-minimum_required_thickness())
df_pc['average_depth']=(df_pc['actual_depth_i']+df_pc['actual_depth_j'])/2
df_pc['average_diameter']=(df_pc['di']+df_pc['dj'])/2

#avg_all_pits_depth
def avg_all_pits_depth():
    return df_pc['average_depth'].mean()

def avg_all_pits_diameter():
    return df_pc['average_diameter'].mean()

def avg_pit_couple_spacing():
    return df_pc['distance'].quantile(0.5)

def mavg():
    mavg=(avg_pit_couple_spacing()-avg_all_pits_diameter())/avg_pit_couple_spacing()
    if mavg<0:
        mavg=0
    return mavg

def eavg():
    return ((3**0.5)*mavg())/2

def RSF():                                                              
    rsf=min(1-(avg_all_pits_depth()/minimum_required_thickness())+((eavg()*(t-future_corrosion_allowance+avg_all_pits_depth()-minimum_required_thickness()))/minimum_required_thickness()),1)
    if rsf<0:
        rsf=0
    return rsf    

def MAWPr():
    return design_cond_pressure*(RSF()/RSFa)

def pit_status():
    if MAWPr()<design_cond_pressure:
        status='UNACCEPTABLE'
    elif MAWPr()>design_cond_pressure: 
        status='ACCEPTABLE'
    return status

#Summary table for Widespread Pitting
dd3 = {'Remaining Strength Factor': [RSF()],
      'Maximum Allowable Working Pressure': [MAWPr()],
      'Overall Pit Status': [pit_status()]
     }
ddf3 = pd.DataFrame(data=dd3)
ddf3=ddf3.style.hide_index()

df_pc['remaining_thickness_ratio_i']=(minimum_required_thickness()-df_pc['actual_depth_i']-future_corrosion_allowance)/minimum_required_thickness()
for i in range(df_pc.index.min(), df_pc.index.max()+1):
    if df_pc['remaining_thickness_ratio_i'].iloc[i]<0:
        df_pc['remaining_thickness_ratio_i'].iloc[i]=0
df_pc['remaining_thickness_ratio_j']=(minimum_required_thickness()-df_pc['actual_depth_j']-future_corrosion_allowance)/minimum_required_thickness()
for i in range(df_pc.index.min(), df_pc.index.max()+1):
    if df_pc['remaining_thickness_ratio_j'].iloc[i]<0:
        df_pc['remaining_thickness_ratio_j'].iloc[i]=0
df_pc['Q_i']=1.123*(((((1-df_pc['remaining_thickness_ratio_i'])/(1-df_pc['remaining_thickness_ratio_i']/RSFa))**2)-1)**0.5)
df_pc['Q_j']=1.123*(((((1-df_pc['remaining_thickness_ratio_j'])/(1-df_pc['remaining_thickness_ratio_j']/RSFa))**2)-1)**0.5)
df_pc['Q_i*sqrt(id*tmin)']=df_pc['Q_i']*((radius_circ()*minimum_required_thickness())**0.5)
df_pc['Q_j*sqrt(id*tmin)']=df_pc['Q_j']*((radius_circ()*minimum_required_thickness())**0.5)
df_pc['cond1_pit_width_i']=(df_pc['wi']<=df_pc['Q_i*sqrt(id*tmin)'])
df_pc['cond1_pit_width_j']=(df_pc['wj']<=df_pc['Q_j*sqrt(id*tmin)'])
df_pc['cond2_pit_depth_i']=(df_pc['di']>=0.2)
df_pc['cond2_pit_depth_j']=(df_pc['dj']>=0.2)




# Dealing with localized pitting
def teq():
    return RSF()*minimum_required_thickness()

def minimum_measured_thickness():
    return teq()

#Only one remaining thickness ratio has to be calculated for LTA, no need to do flaw to flaw spacing
def remaining_thickness_ratio():
    rt=((minimum_measured_thickness()-future_corrosion_allowance)/minimum_required_thickness())
    if rt<0:
        rt=0
    return rt

# S is the width 
def s_local():
    return df_cluster_info['xmax'].max()-df_cluster_info['xmin'].min()
#s_local()

# C is the height
def c_local():
    return df_cluster_info['ymax'].max()-df_cluster_info['ymin'].min()



####### MISSING CODE





# MAPS
df_ll=df_work[df_work['problem_class']=='LOCAL_LOSS'].reset_index().drop('index', 1)
df_pitting=df_work[df_work['problem_class']=='PITTING'].reset_index().drop('index', 1)

my_pal = {"EXTREME": "red", 'CRITICAL_LOSS': 'orange', 'SUBSTANTIAL_LOSS':'yellow', 'WALL_LOSS':'lightgreen', 'PRISTINE':'green', 'none':'white'}

def local_loss_map():
    fig, ax = plt.subplots()
    for i in range(df_ll.index.min(),df_ll.index.max()+1):
        circle1=plt.Circle((df_ll['xcenter'].iloc[i], df_ll['ycenter'].iloc[i]), max(df_ll['s'].iloc[i]/2,df_ll['c'].iloc[i]/2)+0.1, color='b', fill=False)
        ax.add_patch(circle1)

    ax.set_xlim(df_ll['x_min_cluster'].min()-5,df_ll['x_max_cluster'].max()+5)
    ax.set_ylim(df_ll['y_min_cluster'].min()-5,df_ll['y_max_cluster'].max()+5)
    ax.set_aspect('equal')
    sns.scatterplot(df_xyz['x'], df_xyz['y'], hue=df_xyz['TARGET'].values, palette=my_pal)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    return plt.show()

def pitting_map():
    fig, ax = plt.subplots()
    for i in range(df_pitting.index.min(),df_pitting.index.max()+1):
        circle1=plt.Circle((df_pitting['xcenter'].iloc[i], df_pitting['ycenter'].iloc[i]), max(df_pitting['s'].iloc[i]/2,df_pitting['c'].iloc[i]/2)+0.1, color='b', fill=False)
        ax.add_patch(circle1)

    ax.set_xlim(df_pitting['x_min_cluster'].min()-5,df_pitting['x_max_cluster'].max()+5)
    ax.set_ylim(df_pitting['y_min_cluster'].min()-5,df_pitting['y_max_cluster'].max()+5)
    ax.set_aspect('equal')
    sns.scatterplot(df_xyz['x'], df_xyz['y'], hue=df_xyz['TARGET'].values, palette=my_pal)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    return plt.show()


st.write("""
# Diagnostics
""")

st.write("""
## AUT General Info
""")
ddf1

ddf2

st.pyplot(block_grid_heatmap(1))

st.write("""
## Local Metal Loss
""")

if not df_ll.empty:
    df_ll
elif df_ll.empty:
    print('No Local Metal Loss')

try:
    st.pyplot(local_loss_map())
except:
    pass

st.write("""
## Pit Couples
""")
df_pc

st.pyplot(pitting_map())

ddf3

st.write("""
## Additional
""")
st.pyplot(prob_graph(1))



    





