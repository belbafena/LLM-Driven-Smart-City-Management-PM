/* STEP 0 – Start CAS Session */

cas casauto;
caslib _all_ assign;

/* STEP 1 – CLEAN TRAFFIC DATA */
data casuser.traffic_clean;
    set casuser.traffic;
/* Convert text datetime → numeric */
    DateTime_num = input(DateTime, anydtdtm.);
    format DateTime_num datetime20.;

/* Extract date and hour */
    Date = datepart(DateTime_num);
    Hour = hour(DateTime_num);
    format Date yymmdd10.;

/* Remove missing values */
    if Vehicles ne . and Junction ne .;
run;

/* STEP 2 – CLEAN TEMPERATURE DATA */
data casuser.temp_clean;
    set casuser.city_temperature;

    /* Remove invalid values */
    if Day < 1 or Day > 31 then delete;
    if Month < 1 or Month > 12 then delete;
    if Year < 1900 then delete;
    if AvgTemperature = -99 then delete;

    /* Create Date */
    Date = mdy(Month, Day, Year);
    format Date yymmdd10.;

    /* Keep only needed variables */
    keep Date AvgTemperature City Country;
run;
/* STEP 3 – CLEAN RAIN DATA */
data casuser.rain_clean;
    set casuser.bucharest_rain;

    /* Convert character date → numeric SAS date */
    Date_num = input(Date, yymmdd10.);
    format Date_num yymmdd10.;

    keep Date_num Rain_mm;
    rename Date_num = Date;
run;

/* STEP 4 – AGGREGATE DATA 
Traffic → Hourly per Junction
*/
proc fedsql sessref=casauto;
create table casuser.traffic_hourly as
select
    Date,
    Hour,
    Junction,
    avg(Vehicles) as AvgVehicles
from casuser.traffic_clean
group by Date, Hour, Junction;
quit;

/* Temperature → Daily */
proc fedsql sessref=casauto;
create table casuser.temp_daily as
select
    Date,
    avg(AvgTemperature) as AvgTemperature
from casuser.temp_clean
group by Date;
quit;
/* Rain → Daily */
proc fedsql sessref=casauto;
create table casuser.rain_daily as
select Date,
       sum(Rain_mm) as Rain_mm
from casuser.rain_clean
group by Date;
quit;

/* STEP 5 – MERGE DATASETS */
proc fedsql sessref=casauto;
create table casuser.smartcity_base as
select
    t.Date,
    t.Hour,
    t.Junction,
    t.AvgVehicles,
    temp.AvgTemperature,
    rain.Rain_mm
from casuser.traffic_hourly t
left join casuser.temp_daily temp
    on t.Date = temp.Date
left join casuser.rain_daily rain
    on t.Date = rain.Date;
quit;
/* Chexk for missing values after join */
proc means data=casuser.smartcity_base n nmiss;
run;

/* STEP 6 – HANDLE MISSING VALUES */
proc stdize data=casuser.smartcity_base
             out=casuser.smartcity_clean
             method=median
             reponly;
    var AvgTemperature;
run;

/* STEP 7 – CREATE TARGET VARIABLE */
proc univariate data=casuser.smartcity_clean noprint;
    var AvgVehicles;
    output out=casuser.pctl pctlpts=75 pctlpre=P_;
run;
proc sql noprint;
select P_75 into :p75 from casuser.pctl;
quit;

data casuser.final_dataset;
    set casuser.smartcity_clean;

    if AvgVehicles > &p75 then do;
        Congestion = 1;
        Congestion_Label = "HighCongestion";
    end;
    else do;
        Congestion = 0;
        Congestion_Label = "NormalCongestion";
    end;
run;

/* check row */
proc sql;
select count(*) from casuser.final_dataset;
quit;
/* Check Row Counts Before and After */
proc sql;
select count(*) as traffic_clean  from casuser.traffic_clean;
select count(*) as traffic_hourly from casuser.traffic_hourly;
select count(*) as smartcity_base from casuser.smartcity_base;
select count(*) as smartcity_clean from casuser.smartcity_clean;
select count(*) as final_dataset  from casuser.final_dataset;
quit;

/* Sort Data First for lag */
/* Copy to WORK */
data work.temp;
    set casuser.final_dataset;
run;

/* Sort in WORK */
proc sort data=work.temp;
    by Junction Date Hour;
run;

/* Copy back to CAS */
data casuser.smartcity_sorted;
    set work.temp;
run;

/* STEP 8 – ADD SMART-CITY FEATURES */
data casuser.smartcity_features;
    set casuser.smartcity_sorted;
    by Junction;

    /* Temperature Conversion */
    TempC = (AvgTemperature - 32) * 5/9;

    /* Time Features */
    DayOfWeek = weekday(Date);
    Month     = month(Date);
    Weekend   = (DayOfWeek in (1,7));
    RushHour = (Hour in (10,11,12,18,19,20,21));

    /* Weather Features */
    RainFlag  = (Rain_mm > 0);
    HeavyRain = (Rain_mm > 5);
    HotDay    = (TempC > 25);

    /* Lag variables */
    LagTraffic = lag(AvgVehicles);
    Lag2 = lag2(AvgVehicles);
    RainLag = lag(Rain_mm);

/* Reset at new junction */
    if first.Junction then do;
        LagTraffic = .;
        Lag2 = .;
        RainLag = .;
end;

/* 3-Hour Moving Average (SAFE) */
    MovingAvg3hr = mean(AvgVehicles, LagTraffic, Lag2);

    /* Interaction Features */
    Rain_Traffic_Interaction = Rain_mm * AvgVehicles;
    Temp_Traffic_Interaction = TempC * AvgVehicles;

    /* Congestion Index */
    CongestionIndex = AvgVehicles / 50;
run;

/* STEP 10 – Some  EDA VISUALIZATIONS to understand the data before gonna modeling*/

/* --- Visualization 1: Daily Congestion Pattern --- */

proc means data=casuser.smartcity_features noprint;
class Hour;
var Congestion;
output out=cong_hour mean=CongestionRate;
run;

proc sgplot data=cong_hour;
series x=Hour y=CongestionRate / markers;
xaxis label="Hour of Day";
yaxis label="Congestion Probability";
title "Daily Congestion Pattern";


/* Visualization 2 – Traffic Persistence */

proc sgplot data=casuser.smartcity_features;
scatter x=LagTraffic y=AvgVehicles / transparency=0.5;
reg x=LagTraffic y=AvgVehicles;
xaxis label="Previous Traffic Volume";
yaxis label="Current Traffic Volume";
title "Traffic Persistence Effect";
run;

/* Visualization 3 – Junction Congestion Risk */

proc means data=casuser.smartcity_features noprint;
class Junction;
var Congestion;
output out=junction_risk mean=CongestionRate;
run;

proc sgplot data=junction_risk;
hbar Junction / response=CongestionRate datalabel;
xaxis label="Congestion Rate";
title "Traffic Congestion Risk by Junction";
run;
/* STEP 9 – Export dataset for modeling in SAS model Studio */
proc export data=casuser.smartcity_features
	outfile="/export/viya/homes/belete_bafena.ageno@stud.fils.upb.ro/cleaned_integrated_datasetPM.csv"
	dbms=csv
	replace;
run;