# Problem 1
* The number of recorded crimes in 2018 is less than that of 2017 (the difference is 1848)
* According to the graph about the primary type of crimes in Chicago below, we can see that in general, the main types of crime remain stable in 2017 and 2018. There is no sharp decrease or increase in any type of crime on this graph with a scale of 100000 in count. Theft is the crime type with the most frequency in both years. Then comes the battery.
  ![fig1](https://github.com/haonen/Markdown-Photos/blob/master/primary_type_count.png?raw=true)
  
* The top 5 commnuity areas that has highest crime frequency are the same in 2017 and 2018. In community area 25, the frequency of crimes and the proportion of crimes are both decreasing in 2018. But in other 4 community areas, those two indicators are increasing in 2018. Community area 8 has the sharpest increase.

![fig2](https://github.com/haonen/Markdown-Photos/blob/master/top%205%20community%20area.JPG?raw=true)
![fig3](https://github.com/haonen/Markdown-Photos/blob/master/comm_area_cnt.png?raw=true)
![fig4](https://github.com/haonen/Markdown-Photos/blob/master/comm_area_pro.png?raw=true)

* The top 5 locations that have highest frequency of crimes occurence remain the same in both years. Street is the most dangerous place if we take the frequency of crimes as an indicator. For the changes in proportions to total crimes, apartments have a higher proportion in 2018 while others have slightly decrease.
![fig5](https://github.com/haonen/Markdown-Photos/blob/master/loc_cnt.png?raw=true)
![fig6](https://github.com/haonen/Markdown-Photos/blob/master/loc_pro.png?raw=true)

# Problem 2
For this problem, in order to save the time of getting zipcode from geocode by using Google Map API, I filter out dataframes for "battery" with locations that have more than 10 crime records. And then I use the Census API (American Facts Finder) and pick some variables on zipcode level from 2013-2017 American Community Survey 5-Year Estimates: unemployment rate, median household income, poverty rate, rate for four main races (white, black, hispanic or latino and asian), family size. I filter out the data frame for ACS data by checking whether these zipcodes are in crime reports. And I analyze the filtered data frames for different crime types and different years to see if it varies in characteristics of locations of different years or in different crime types.  
To avoid the influences of outliners, I choose median as the statistics to describe block/zipcode characteristics. 
For the reference of the following analysis, I record the data of those variables on Chicago City level from the same survey result:

<table>
  <tr>
    <th>unemployment rate</th>
    <th>median household income</th>
    <th>poverty rate</th>
    <th>white rate</th>
    <th>black rate</th>
    <th>hispanic rate</th>
    <th>asian rate</th>
    <th>average family size</th>    
  </tr>
  <tr>
    <td>9.9</td>
    <td>52,497</td>
    <td>20.6</td>
    <td>51.2</td>
    <td>31.6</td>
    <td>29.0</td>
    <td>7.2</td>
    <td>3.4</td>
  </tr>
</table>

## 1.  
<img src="https://github.com/haonen/Markdown-Photos/blob/master/median_battery.JPG?raw=true" alt="fig7" width="800"/>  

The zipcodes/blocks that have high frequency of battery have the following characteristics. The unemployment rate is around 5.7% and the poverty rate is around 20%. The median household income is around 50000 dollars and the average family size is around 3 people. The marjority of their population is the white and then comes the black. 

## 2.
<img src="https://github.com/haonen/Markdown-Photos/blob/master/median_homicide.JPG?raw=true" alt="fig7" width="800"/>   

The zipcodes/blocks that have high frequency of homicide incidents have the following characteristics. The unemployment rate is around 6.3% and the poverty rate is around 20%. The median household income is also around 50000 dollars and the average family size is around 3 people. The marjority of their population is the white and then comes the black.    

Both types of crime would happen in areas that have roughly the same poverty rate as Chicago city and lower median household income. But their unemployment rate are lower than the city situation and their population composition is more diverse.    

## 3. 
For battery, based on the tables above and the radar chart below, we can see that, the most obvious variation is population composition. Both rates of the white and the hispanic or latino increase from 2017 to 2018. Besides that, the tables tell us that there is an increase in the median household income, but we cannot tell whether it is due to inflation. The poverty rate decrease slightly and the familiy size increases.    

![fig9](https://github.com/haonen/Markdown-Photos/blob/master/picture_battery.png?raw=true)   

The report for homicide has a similar situation. The proportion of white people increase for these areas. The unemployment rate remains constant while the median household income also increases. Poverty rates decreases slightly and average family size increases a little bit.

![fig10](https://github.com/haonen/Markdown-Photos/blob/master/picture_homicide.png?raw=true)   

## 4.
We only use the data for 2017 here.
<img src="https://github.com/haonen/Markdown-Photos/blob/master/compare_so_dp.JPG?raw=true" alt="drawing" width="800"/>  
<img src="https://github.com/haonen/Markdown-Photos/blob/master/so_dp.JPG?raw=true" alt="drawing" width="400"/>   
According to the radar chart above, the blocks that get “Deceptive Practice” have more proportion of white population, less proportion of black population and slightly smaller proportion of hispanic or latino population than the blocks that get “Sex Offense". Also, their poverty rate is lower. The averge household income is higher in  the blocks that get “Deceptive Practice”. 

# Problem 3  
## 1.
From 2017 to 2018, the main types of crime do not change a lot. Some of their numbers decrease but some of them increase. And we cannot say for sure that there is a good or a bad change because in general, the records number in 2018 is less than the records number in 2017. The community areas that have high frequency of crimes happen also remain stable.    
Those "crime blocks" above mainly have much lower median household income than city level and competitive poverty rate. But there is an increase in median household income and a decrease in poverty rate. However, their unemployment rates are much lower than city level and their population composition are more diverse. We can also see an increase in the proportion of white people in those blocks.
## 2.
### A
To avoid the influence of randomly reports number in different years, I calculate the increase percent based on the proprotion of different types of crimes for this part.   
These statistics are not correct. Here is my output:   
 ```
Robbery increase:-16.731%
Battery increase:4.895%
Burglary increase:-11.464%
Motor vehicle theft increase:-11.461%
 ```
 I guess that they might mistake the formula for calculating increase in percentage. For example, if they take the number of records in 2018 as denominator and mistake the sign, then the percentage increase would be roughly 21% (but actually it is decreasing).   
 ### B
These results could be misleading and I do not aggree with his conclusion. Because the choose of time frame is arbitrary and if we pick different time frame, the result could be inconsistent. For instance, here is the output for July:
```
Robbery increase:-12.143%
Battery increase:4.527%
Burglary increase:-4.9%
Motor vehicle theft increase:-12.109%
```
And the result for July 26th:
```
Robbery increase:26.8%
Battery increase:0.694%
Burglary increase:-10.251%
Motor vehicle theft increase:-23.611%
```
For the comparison of July, the robbery decrease but for the particular day, the robbery actually increase a lot. And for Battery, it increase for the time frame of July but remain constant for particular day. So, if we pick different time frame, we might have different conclusions. And we cannot ensure that the time frame we pick guarantee roughly the same enviornment or context for crimes in both years.   
## 3
* There is an increase in the proportion of white population in those areas that have high frequency of "homicide" and "battery". This might be a side effects of gentrification in some communities of Chicago since those areas originally have diverse population. Pay attention to the gentrification.  
* Since those area have diverse population, I would suggest the mayor to hear more from minorities and check out their living enviornment to see what cause they live in the danger of crimes.   
* The unemployment rate is much lower than city level but median household income is much lower than the city level. This might indicate that albeit people do get jobs in those areas, they earn little and it could also be a source of dissatisfaction and causes violence and crimes. The City of Chicago should focus on low income groups.   
* The poverty rate in those area are roughly the same as city level and it decreases at a slow speed. Poverty can be a source of violence and crime, so the mayor should pay attention to reduce the poverty rate.
* Theft is a really sever type of crimes in Chicago and it has an increase in proprotion . I would suggest the city government to pay attention to this high frequency.   
## 4
I would provide the caveats saying that since all of my analysis above is based on descriptive statistics rather than statistical inference, most of my recommendation are based on my guess. The true correlation and causality still needs further study.   

# Problem 4
## A
2111 S Michigan Ave is in the Near South Side (Community Aread code is 33) of Chicago. I first filter out all the records from this community and then count the value of each crime type. Divide the number of records for each type by the length of all the records from Near South Side, I can find out the the probabilities for each type of request. And theft is the one with highest probability.   
```
THEFT                               28.822
DECEPTIVE PRACTICE                  15.651
BATTERY                             15.496
CRIMINAL DAMAGE                      8.006
OTHER OFFENSE                        5.940
ASSAULT                              4.855
ROBBERY                              4.804
MOTOR VEHICLE THEFT                  4.649
CRIMINAL TRESPASS                    4.390
BURGLARY                             3.151
NARCOTICS                            0.930
SEX OFFENSE                          0.775
OFFENSE INVOLVING CHILDREN           0.568
CRIM SEXUAL ASSAULT                  0.465
WEAPONS VIOLATION                    0.413
PROSTITUTION                         0.310
PUBLIC PEACE VIOLATION               0.207
INTERFERENCE WITH PUBLIC OFFICER     0.207
STALKING                             0.103
OBSCENITY                            0.103
NON-CRIMINAL                         0.103
ARSON                                0.052
```

## B
The commnuity area code for Uptown is 3 and for Garfield Park is 26 and 17. For each year report, I first filter out the theft records and then filter out the records happened in Uptown or Garfield Park. Dividing the value counts for each community area by the length of theft records for each year, I can come up with the estimates of probability. 
<table>
  <tr>
    <th>Community Area</th>
    <th>Probability 2017</th>
    <th>Probability 2018</th>
  </tr>
  <tr>
    <td>Uptown</td>
    <td>0.015044</td>
    <td>0.015151</td>
  </tr>
  <tr>
    <td>Garfield Park</td>
    <td>0.009061 + 0.008983 = 0.018044</td>
    <td>0.010802 + 0.009681 = 0.117701</td>
  </tr>
</table>


In 2017, it is more likely that this call comes from Garfield Park and it has 0.018044 - 0.015044 = 0.003 higher probability.   
In 2018, albeit the theft happended in both communities rorse, it is still more likely that this call comes from  Garfield Park and it has 0.117701 - 0.015151 = 0.10255 higher probability.   
  
## C
According to Bayes' theorem:   
P(Battery | Garfield Park) = P(Battery) * P(Garfield Park | Battery)/P(Garfield Park) = ((100+160)/1000 * (100/100+160))/(600/1000)=1/6   
P(Battery | Uptown) = P(Battery) * P(Uptown | Battery)/P(Uptown) = 2/5   
2/5-1/6 = 0.23   
So, the probability that this call comes from Uptown is roughly 23% higher than the probability of it coming from Garfield Park.
