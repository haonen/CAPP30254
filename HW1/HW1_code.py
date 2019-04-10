"""
CAPP30254 HW1

YUWEI ZHANG
"""

import pandas as pd
import numpy as np
import seaborn as sns
from math import pi
import re
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import googlemaps

COLTYPE = {'Community Area': str}

#############
# Load Data #
#############
crimes_2017 = pd.read_csv('Crimes_2017.csv',
                          dtype=COLTYPE)
crimes_2017.head()
crimes_2018 = pd.read_csv('Crimes_2018.csv',
                          dtype=COLTYPE)
crimes_2018.head()
crimes = crimes_2017.append(crimes_2018)

##############
# Problem 1  #
##############

#differences in reports length
len(crimes_2017)-len(crimes_2018)

#Descriptive analysis for crime types
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
with sns.color_palette("BuGn_r"):
    sns.countplot(y="Primary Type", hue="Year", ax=ax, data=crimes)
fig.savefig('primary_type_count.png')

#Descriptive analysis for communuity area
comm_2017 = crimes_2017['Community Area'].value_counts().to_frame()
comm_2017['proportion'] = round(comm_2017['Community Area'] /
                                comm_2017['Community Area'].sum() * 100, 3)
head_comm_2017 = comm_2017.head()
head_comm_2017 = head_comm_2017.reset_index()
head_comm_2017 = head_comm_2017.rename(columns={'index': 'community area',
                                                'Community Area': 'cnt'})

comm_2018 = crimes_2018['Community Area'].value_counts().to_frame()
comm_2018['proportion'] = round(comm_2018['Community Area'] /
                                comm_2017['Community Area'].sum() * 100, 3)
head_comm_2018 = comm_2018.head()
head_comm_2018 = head_comm_2018.reset_index()
head_comm_2018 = head_comm_2018.rename(columns={'index': 'community area',
                                                'Community Area': 'cnt'})

head_comm_2017['year'] = 2017
head_comm_2018['year'] = 2018
head_comm = head_comm_2017.append(head_comm_2018)
fig1 = sns.barplot(x='community area', y='cnt', hue='year', data=head_comm)
fig1 = fig1.get_figure()
fig1.savefig('comm_area_cnt.png')

fig2 = sns.barplot(x='community area', y='proportion', hue='year',
                   data=head_comm)
fig2 = fig2.get_figure()
fig2.savefig('comm_area_pro.png')

#Descriptive analysis for location description
loc_des_2017 = crimes_2017['Location Description'].value_counts().to_frame()
loc_des_2017['proportion'] = round(loc_des_2017['Location Description'] /
                                   loc_des_2017['Location Description'].sum()
                                   * 100, 3)
head_loc_2017 = loc_des_2017.head()
head_loc_2017 = head_loc_2017.reset_index()
head_loc_2017 = head_loc_2017.rename(columns={'index': 'location description',
                                      'Location Description': 'cnt'})

loc_des_2018 = crimes_2018['Location Description'].value_counts().to_frame()
loc_des_2018['proportion'] = round(loc_des_2018['Location Description'] /
                                   loc_des_2017['Location Description'].sum()
                                   * 100, 3)
head_loc_2018 = loc_des_2018.head()
head_loc_2018 = head_loc_2018.reset_index()
head_loc_2018 = head_loc_2018.rename(columns={'index': 'location description',
                                      'Location Description': 'cnt'})

head_loc_2017['year'] = 2017
head_loc_2018['year'] = 2018
head_loc = head_loc_2017.append(head_loc_2018)
fig1 = sns.barplot(x='location description', y='cnt', hue='year',data=head_loc)
fig1 = fig1.get_figure()
fig1.savefig('loc_cnt.png')

fig2 = sns.barplot(x='location description', y='proportion', hue='year',
                   data=head_loc)
fig2 = fig2.get_figure()
fig2.savefig('loc_pro.png')

##############
# Problem 2  #
##############
# filter dataframe
coltype = {'unemployment rate': float, 'poverty rate': float}
acs_2017 = pd.read_csv('acs_data.csv', dtype=coltype)

battery_2017  =  crimes_2017[crimes_2017['Primary Type'] == 'BATTERY']
battery_2018  =  crimes_2018[crimes_2018['Primary Type'] == 'BATTERY']
homicide_2017  =  crimes_2017[crimes_2017['Primary Type'] == 'HOMICIDE']
homicide_2018  =  crimes_2018[crimes_2018['Primary Type'] == 'HOMICIDE']
DP_2017 = crimes_2017[crimes_2017['Primary Type'] == 'DECEPTIVE PRACTICE']
SO_2017 = crimes_2017[crimes_2017['Primary Type'] == 'SEX OFFENSE']

#get zipcode through latitude and longitude (using google map api)
gmaps_key = googlemaps.Client(key='AIzaSyABYmQbCm2aro_JDQkTV_Td96fvUA6g_nY')

def pick_zipcode(row):
    '''
    check whether the formated address in index 0 or index 1 has the zipcode
    Input:
        a row
    Return:
        zipcode(str)
    '''
    match = re.search(r"(IL )([0-9]{5})", row[0]['formatted_address'])
    if not match:
        match = re.search(r"(IL )([0-9]{5})", row[1]['formatted_address'])
    return match.group(2)


def get_zipcode(df):
    '''
    Get zipcode based on the geocode information
    Input:
        a dataframe
    Return:
        a new dataframe with zipcode column
    '''
    df['Location'] = df.apply(lambda x: (x['Latitude'], x['Longitude'])
                              if not pd.isnull(x['Location'])
                              else np.nan, axis=1)
    df['geocode_result'] = df.apply(lambda x: gmaps_key.reverse_geocode(
                                    x['Location']) if not pd.isnull(
                                    x['Location']) else None, axis = 1)
    df['zipcode'] = df.apply(lambda x: pick_zipcode(x['geocode_result'])
                             if x['geocode_result'] else np.nan, axis=1)
    df['zipcode'] = df['zipcode'].fillna(0).astype(np.int64)
    return df


#create median table
def create_median_table(df1, df2, var_list=['unemployment rate',\
                                      'median household income',\
                                      'poverty rate', 'white rate',\
                                     'black rate', 'hispanic rate',\
                                     'asian rate', 'average family size']):
    '''
    create a table for the median statistics for two years
    Inputs:
        two dataframes
        a variable list
    Return:
        the median data frame
    '''
    median_dict = {}
    for var in var_list:
        median_dict[var] = [df1[var].median(), df2[var].median()]
    new_df = pd.DataFrame(data=median_dict)
    new_df.insert(0, "year", [2017, 2018])
    return new_df


#Plot radar charts
def create_radar(group1, group2, df):
    '''
    create radar chart
    Inpust:
        the name of two groups
        a data frame that contains the information about five variables
    Output:
        create and save the radar chart
    '''
    # PART 1: Create background
    categories=list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30,40,50,60], ["10","20","30","40","50","60"], color="grey", size=7)
    plt.ylim(0,60)
    # PART 2: Add plots
    # group1
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=group1)
    ax.fill(angles, values, 'b', alpha=0.1)
    # group2
    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=group2)
    ax.fill(angles, values, 'r', alpha=0.1)
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(-0.1, 0.1))
   
    plt.savefig('picture.png'， bbox_inches='tight')
    
# Part 1 & Part 3
def high_loc(df):
    '''
    filter the df with high frequency of crimes and save its geocode
    Input:
        a data frame
    Return:
        a new data frame only contains locations  having high frequency of 
        battery
    '''
    new_df = df['Location'].value_counts().to_frame()
    new_df = new_df[new_df['Location'] >= 10]
    new_df = new_df.reset_index()
    new_df = new_df.rename(columns={'index': 'Location', 'Location': 'cnt'})
    new_df['Latitude'] = new_df.apply(lambda x: re.search((r'(\d\d\.[0-9]+)(\,)'),\
                                                      x['Location']).group(1)\
                                                      if not (x['Location']\
                                                      is np.nan) else np.nan,\
                                                      axis=1)
    new_df['Longitude'] = new_df.apply(lambda x: re.search((r'(\, )(-\d\d\.[0-9]+)'),\
                                                        x['Location']).group(2)\
                                                        if not (x['Location']\
                                                        is np.nan) else np.nan,\
                                                        axis=1)
    return new_df


high_loc_2017 = high_loc(battery_2017)
high_loc_2018 = high_loc(battery_2018)

zipcode_battery_2017 = get_zipcode(high_loc_2017)
zipcode_battery_2018 = get_zipcode(high_loc_2018)

battery_2017_filtered = acs_2017[acs_2017['zipcode'].isin(zipcode_battery_2017['zipcode'])]
battery_2018_filtered = acs_2017[acs_2017['zipcode'].isin(zipcode_battery_2018['zipcode'])]

median_df_battery = create_median_table(battery_2017_filtered, battery_2018_filtered)
median_df_battery

df = pd.DataFrame({
'group': ['battery_2017','battery_2018'],
'unemployment rate': [5.8, 5.7],
'povertry rate': [21.2, 20.5],
'white rate': [31.3, 34.0],
'black rate': [17.5, 17.8],
'hispanic rate': [8.7, 11.8]
})

create_radar('battery_2017', 'battery_2018', df)

# Part 2 & Part3
zipcode_homicide_2017 = get_zipcode(homicide_2017)
zipcode_homicide_2018 = get_zipcode(homicide_2018)

homicide_2017_filtered = acs_2017[acs_2017['zipcode'].isin(zipcode_homicide_2017['zipcode'])]
homicide_2018_filtered = acs_2017[acs_2017['zipcode'].isin(zipcode_homicide_2018['zipcode'])]

median_df_homicide = create_median_table(homicide_2017_filtered, homicide_2018_filtered)
median_df_homicide.head()

df = pd.DataFrame({
'group': ['homicide_2017','homicide_2018'],
'unemployment rate': [6.3, 6.3],
'povertry rate': [20.8, 20.7],
'white rate': [25.5, 28.2],
'black rate': [17.9, 17.8],
'hispanic rate': [13.2, 13.3]
})

create_radar('homicide_2017', 'homicide_2018', df)

# Part 4
zipcode_SO_2017 = get_zipcode(SO_2017)
SO_2017_filtered = acs_2017[acs_2017['zipcode'].isin(
                            zipcode_SO_2017['zipcode'])]

zipcode_DP_2017 = get_zipcode(DP_2017)
DP_2017_filtered = acs_2017[acs_2017['zipcode'].isin(zipcode_DP_2017)]

df = pd.DataFrame({
'group': ['Deceptive Practice','Sex Offense'],
'unemployment rate': [4.9, 5.1],
'povertry rate': [15.0, 18.8],
'white rate': [51.3, 43.5],
'black rate': [10.9, 15.2],
'hispanic rate': [12.5, 11.9]
})
    
create_radar('Deceptive Practice', 'Sex Offense', df)


##############
# Problem 3  #
##############

# Part 2
def get_july_record(df, crime, pattern):
    '''
    get the records for certain time peroid of certain crime
    Inputs:
        a data frame
        a crime type(str)
        a re pattern(to specify the time frame)
    Return:
        a new data frame
    '''
    new_df = df[df['Primary Type'] == crime]
    new_df['match'] = new_df.apply(lambda x: bool(re.findall(pattern,
                                   x['Date'])), axis=1)
    july_df = new_df[new_df['match'] == True]
    return july_df


def compute_increase(df1, df2, crime, pattern):
    '''
    compute the percentage increase of proportion
    Inputs:
        two data frame for different years
        a crime type
        a re pattern
    Return:
        the percentage increase(float)
    '''
    july_df1 = get_july_record(df1, crime, pattern)
    july_df2 = get_july_record(df2, crime, pattern)
     return ((len(july_df2) / len(df2) - len(july_df1) / len(df1)) /
             (len(july_df1) / len(df1)) * 100)
     
#Check the week before July 26th
pattern = r'07/25|07/24|07/23|07/22|07/21|07/20|07/19'
print("Robbery increase:{}%".format(round(compute_increase(crimes_2017, 
                                    crimes_2018, 'ROBBERY', pattern), 3)))
print("Battery increase:{}%".format(round(compute_increase(crimes_2017,
                                    crimes_2018, 'BATTERY', pattern), 3)))
print("Burglary increase:{}%".format(round(compute_increase(crimes_2017,
                                     crimes_2018, 'BURGLARY', pattern), 3)))
print("Motor vehicle theft increase:{}%".format(round(compute_increase(
    crimes_2017, crimes_2018, 'MOTOR VEHICLE THEFT', pattern), 3)))

#Check July
pattern = r'^07/'
print("Robbery increase:{}%".format(round(compute_increase(crimes_2017, 
                                    crimes_2018, 'ROBBERY', pattern), 3)))
print("Battery increase:{}%".format(round(compute_increase(crimes_2017,
                                    crimes_2018, 'BATTERY', pattern), 3)))
print("Burglary increase:{}%".format(round(compute_increase(crimes_2017,
                                     crimes_2018, 'BURGLARY', pattern), 3)))
print("Motor vehicle theft increase:{}%".format(round(compute_increase(
    crimes_2017, crimes_2018, 'MOTOR VEHICLE THEFT', pattern), 3)))
#check July 26th
pattern = r'07/26'
print("Robbery increase:{}%".format(round(compute_increase(crimes_2017, 
                                    crimes_2018, 'ROBBERY', pattern), 3)))
print("Battery increase:{}%".format(round(compute_increase(crimes_2017,
                                    crimes_2018, 'BATTERY', pattern), 3)))
print("Burglary increase:{}%".format(round(compute_increase(crimes_2017,
                                     crimes_2018, 'BURGLARY', pattern), 3)))
print("Motor vehicle theft increase:{}%".format(round(compute_increase(
    crimes_2017, crimes_2018, 'MOTOR VEHICLE THEFT', pattern), 3)))


##############
# Problem 4  #
##############

#Part A
NSS = crimes_2017[crimes_2017['Community Area'] == 33]
round(NSS['Primary Type'].value_counts()/len(NSS) * 100, 3)

#Part B
theft_2017 = crimes_2017[crimes_2017['Primary Type'] == 'THEFT']
filter_condition = ((theft_2017['Community Area'] == 3) |
                    (theft_2017['Community Area'] == 26) |
                    (theft_2017['Community Area'] == 27))
theft_2017[filter_condition]['Community Area'].value_counts()/len(theft_2017)

theft_2018 = crimes_2018[crimes_2018['Primary Type'] == 'THEFT']
filter_condition = ((theft_2018['Community Area'] == 3) |
                    (theft_2018['Community Area'] == 26) |
                    (theft_2018['Community Area'] == 27))
theft_2018[filter_condition]['Community Area'].value_counts()/len(theft_2018)
