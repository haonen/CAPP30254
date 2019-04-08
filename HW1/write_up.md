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
For this problem, in order to save the time of getting zipcode from geocode by using Google Map API, I filter out dataframes for the four crime types specified in questions with locations that have more than 10 crime records. And then I use the Census API (American Facts Finder) and pick some variables on zipcode level from 2013-2017 American Community Survey 5-Year Estimates: unemployment rate, median household income, poverty rate, rate for four main races (white, black, hispanic or latino and asian), family size. I filter out the data frame for ACS data by checking whether these zipcodes are in crime reports. And I analyze the filtered data frames for different crime types and different years to see if it varies in characteristics of locations of different years or in different crime types.
## 1.
!fig[7]
## 2.
!fig[8]
## 3. 
![fig9](https://github.com/haonen/Markdown-Photos/blob/master/picture_battery.png?raw=true)

![fig10](https://github.com/haonen/Markdown-Photos/blob/master/picture_homicide.png?raw=true)
## 4.
<img src="https://github.com/haonen/Markdown-Photos/blob/master/so_dp.JPG?raw=true" alt="drawing" width="400"/>

# Problem 3  
## 1.
## 2.
### A
These statistics are not correct. Here is my output:
 ```
Robbery increase:-17.304%
Battery increase:4.172%
Burglary increase:-12.074%
Motor vehicle theft increase:-12.072%
 ```
 I guess that they might mistake the formula for calculating increase in percentage. For example, if they take the number of records in 2018 as denominator and mistake the sign, then the percentage increase would be roughly 21% (but actually it is decreasing).   
 ### B
These results could be misleading and I do not aggree with his conclusion. Because the choose of time frame is arbitrary and if we pick different time frame, the result could be inconsistent. For instance, here is the output for July:
```
Robbery increase:-12.749%
Battery increase:3.807%
Burglary increase:-5.556%
Motor vehicle theft increase:-12.714%
```
And the result for July 26th:
```
Robbery increase:25.926%
Battery increase:0.0%
Burglary increase:-10.87%
Motor vehicle theft increase:-24.138%
```
For the comparison of July, the robbery decrease but for the particular day, the robbery actually increase a lot. And for Battery, it increase for the time frame of July but remain constant for particular day. So, if we pick different time frame, we might have different conclusions. And we cannot ensure that the time frame we pick guarantee roughly the same enviornment or context for crimes in both years.   
## 3
## 4


# Problem 4
## A
2111 S Michigan Ave is in the Near South Side (Community Aread code is 33) of Chicago. I first filter out all the records from this community and then count the value of each crime type. Divide the number of records for each type by the length of all the records from Near South Side, I can find out the the probabilities for each type of request. And theft is the one with highest probability.   
![figp4a](https://github.com/haonen/Markdown-Photos/blob/master/p4a.JPG?raw=true)

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
