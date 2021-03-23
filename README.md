# Metis Data Science Bootcamp | Project 5

---

## **Investigation of US Traffic Accidents and Prediction of Accident Severity**

**Classification using neural network**

Project timeline: 3 weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf).

------------

### **Introduction** 

Traffic accidents are a leading cause of death in the USA for adults and a leading cause of nonnatual death for US citizens. There were 33,244 fatal motor vehicle crashes in the United States in 2019 in which 36,096 deaths occurred. This resulted in 11.0 deaths per 100,000 people and 1.11 deaths per 100 million miles traveled [(iihs.org)](https://www.iihs.org/topics/fatality-statistics/detail/state-by-state). An additional 4.4 million are injured seriously enough to require medical attention. All the evidence suggests that the U.S. suffers the most road crash deaths of any high-income country, about 50% higher than similar countries in Western Europe, Canada, Australia and Japan [(asirt.org)](https://www.asirt.org/safe-travel/road-safety-facts/). Therefore, it is urgent to understand the underlying mechanisms of the occurrence of traffic accidents. This study aimed to investigate the relevance of accident occurrence to time, name of day, season, and weather conditions and to build a neural network for instantaneous prediction of accident severity.  

***********************

### **Methods**

#### Data acquisition

The dataset was downloaded from ([Moosavi et al. (2019a)](https://arxiv.org/pdf/1909.09638.pdf); [Moosavi et al. (2019b)](https://arxiv.org/pdf/1906.05409.pdf)), where the authors collected the data from MapQuest and bing and are continuously updating the dataset. The dataset includes about 4.2 million traffic accidents which covers 49 states of the USA. All the features are listed below [(REF)](https://smoosavi.org/datasets/us_accidents). 

|  #   |       Attribute       | Description                                                  | Nullable |
| :--: | :-------------------: | :----------------------------------------------------------- | :------: |
|  1   |          ID           | This is a unique identifier of the accident record.          |    No    |
|  2   |        Source         | Indicates source of the accident report (i.e. the API which reported the accident.). |    No    |
|  3   |          TMC          | A traffic accident may have a [Traffic Message Channel (TMC)](https://wiki.openstreetmap.org/wiki/TMC/Event_Code_List) code which provides more detailed description of the event. |   Yes    |
|  4   |       Severity        | Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay). |    No    |
|  5   |      Start_Time       | Shows start time of the accident in local time zone.         |    No    |
|  6   |       End_Time        | Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow was dismissed. |    No    |
|  7   |       Start_Lat       | Shows latitude in GPS coordinate of the start point.         |    No    |
|  8   |       Start_Lng       | Shows longitude in GPS coordinate of the start point.        |    No    |
|  9   |        End_Lat        | Shows latitude in GPS coordinate of the end point.           |   Yes    |
|  10  |        End_Lng        | Shows longitude in GPS coordinate of the end point.          |   Yes    |
|  11  |     Distance(mi)      | The length of the road extent affected by the accident.      |    No    |
|  12  |      Description      | Shows natural language description of the accident.          |    No    |
|  13  |        Number         | Shows the street number in address field.                    |   Yes    |
|  14  |        Street         | Shows the street name in address field.                      |   Yes    |
|  15  |         Side          | Shows the relative side of the street (Right/Left) in address field. |   Yes    |
|  16  |         City          | Shows the city in address field.                             |   Yes    |
|  17  |        County         | Shows the county in address field.                           |   Yes    |
|  18  |         State         | Shows the state in address field.                            |   Yes    |
|  19  |        Zipcode        | Shows the zipcode in address field.                          |   Yes    |
|  20  |        Country        | Shows the country in address field.                          |   Yes    |
|  21  |       Timezone        | Shows timezone based on the location of the accident (eastern, central, etc.). |   Yes    |
|  22  |     Airport_Code      | Denotes an airport-based weather station which is the closest one to location of the accident. |   Yes    |
|  23  |   Weather_Timestamp   | Shows the time-stamp of weather observation record (in local time). |   Yes    |
|  24  |    Temperature(F)     | Shows the temperature (in Fahrenheit).                       |   Yes    |
|  25  |     Wind_Chill(F)     | Shows the wind chill (in Fahrenheit).                        |   Yes    |
|  26  |      Humidity(%)      | Shows the humidity (in percentage).                          |   Yes    |
|  27  |     Pressure(in)      | Shows the air pressure (in inches).                          |   Yes    |
|  28  |    Visibility(mi)     | Shows visibility (in miles).                                 |   Yes    |
|  29  |    Wind_Direction     | Shows wind direction.                                        |   Yes    |
|  30  |    Wind_Speed(mph)    | Shows wind speed (in miles per hour).                        |   Yes    |
|  31  |   Precipitation(in)   | Shows precipitation amount in inches, if there is any.       |   Yes    |
|  32  |   Weather_Condition   | Shows the weather condition (rain, snow, thunderstorm, fog, etc.) |   Yes    |
|  33  |        Amenity        | A [POI](https://wiki.openstreetmap.org/wiki/Points_of_interest) annotation which indicates presence of [amenity](https://wiki.openstreetmap.org/wiki/Key:amenity) in a nearby location. |    No    |
|  34  |         Bump          | A POI annotation which indicates presence of speed bump or hump in a nearby location. |    No    |
|  35  |       Crossing        | A POI annotation which indicates presence of [crossing](https://wiki.openstreetmap.org/wiki/Key:crossing) in a nearby location. |    No    |
|  36  |       Give_Way        | A POI annotation which indicates presence of [give_way](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dgive_way) in a nearby location. |    No    |
|  37  |       Junction        | A POI annotation which indicates presence of [junction](https://wiki.openstreetmap.org/wiki/Key:junction) in a nearby location. |    No    |
|  38  |        No_Exit        | A POI annotation which indicates presence of [no_exit](https://wiki.openstreetmap.org/wiki/Key:noexit) in a nearby location. |    No    |
|  39  |        Railway        | A POI annotation which indicates presence of [railway](https://wiki.openstreetmap.org/wiki/Key:railway) in a nearby location. |    No    |
|  40  |      Roundabout       | A POI annotation which indicates presence of [roundabout](https://wiki.openstreetmap.org/wiki/Tag:junction%3Droundabout) in a nearby location. |    No    |
|  41  |        Station        | A POI annotation which indicates presence of [station](https://wiki.openstreetmap.org/wiki/Key:station) in a nearby location. |    No    |
|  42  |         Stop          | A POI annotation which indicates presence of [stop](https://wiki.openstreetmap.org/wiki/Key:stop) in a nearby location. |    No    |
|  43  |    Traffic_Calming    | A POI annotation which indicates presence of [traffic_calming](https://wiki.openstreetmap.org/wiki/Key:traffic_calming) in a nearby location. |    No    |
|  44  |    Traffic_Signal     | A POI annotation which indicates presence of [traffic_signal](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dtraffic_signals) in a nearby location. |    No    |
|  45  |     Turning_Loop      | A POI annotation which indicates presence of [turning_loop](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dturning_loop) in a nearby location. |    No    |
|  46  |    Sunrise_Sunset     | Shows the period of day (i.e. day or night) based on sunrise/sunset. |   Yes    |
|  47  |    Civil_Twilight     | Shows the period of day (i.e. day or night) based on [civil twilight](https://en.wikipedia.org/wiki/Twilight#Civil_twilight). |   Yes    |
|  48  |   Nautical_Twilight   | Shows the period of day (i.e. day or night) based on [nautical twilight](https://en.wikipedia.org/wiki/Twilight#Nautical_twilight). |   Yes    |
|  49  | Astronomical_Twilight | Shows the period of day (i.e. day or night) based on [astronomical twilight](https://en.wikipedia.org/wiki/Twilight#Astronomical_twilight). |   Yes    |

#### Data cleaning and preprocessing

All the accidents with missing values, regardless of features, were excluded to avoid the influence of manual operation on the performance of the neural network.  

The start time of accidents was converted to representative time periods, i.e. early morning, morning, afternoon, evening, night, and late night. The date and month when each accident occurred were converted to the name of day and season, respectively. 

Because various weather conditions were reported in the data, only the most common weather conditions were considered. They included clear, fair, cloudy, windy, rain, snow, obscuration, and sand storm. Wind directions were simplified to north, south, west, east, northeast, northwest, southeast, and southwest.

For the point of interest, the features with very limited number of samples were excluded. First, the percentage of "True" values for each point of interest feature was calculated. Next, a threshold using the 75th percentile of all these percentage values of point of interest features were determined. Last, the point of interest features whose percentage of "True" values was smaller than the threshold were excluded. The final retained features included "Crossing", "Junction", "Traffic_Signal".

All other useless columns were removed. They included ID, Source, TMC, Start_Time, End_Time, End_Lat, End_Lng, Number, Street, City, County, Country, Zipcode, Timezone, Airport_Code, Wind_Chill(f),  Turning_Loop, Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_twilight, Weather_Timestamp, Description.









#### Exploratory analysis

First, for each severity level and state, the number of accidents was counted and divided by the total number of accidents for the corresponding severity to calculate the accident rate for each state. Second, 

#### Featuring engineering

xx



---

### Results and Discussion

XX





---

### Conclusions

XX







