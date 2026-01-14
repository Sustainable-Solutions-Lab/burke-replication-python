# Figure 2 Code Comparison: Original Stata/R vs Python Translation

This document contains the code for generating Figure 2 from Burke, Hsiang, and Miguel (2015).
The goal is to verify that the Python translation correctly replicates the original Stata/R methodology,
particularly for computing confidence intervals.

## QUESTION FOR ANALYSIS

The Python translation produces error bars that don't match the published paper's Figure 2.
Please analyze whether the Python code correctly replicates the Stata/R logic, especially:

1. How Stata's `margins` command computes predictions and standard errors
2. How the R code normalizes by subtracting the maximum
3. Whether the Python delta method approach is equivalent

---

## PART 1: ORIGINAL STATA CODE (GenerateFigure2Data.do)

This Stata code generates the data that R then plots.

```stata
* RESULTS TO CONSTRUCT RESPONSE FUNCTIONS IN FIGURE 2
//		Note: users will need to install parmest command (e.g. "ssc install parmest")

clear all
set mem 1G
set matsize 10000
set maxvar 10000
set more off

// ---------- ESTIMATE GLOBAL RESPONSE WITH BASELINE REGRESSION SPECIFICATION
// ---------- THEN WRITE OUT FOR CONSTUCTION OF FIGURE 2, panel A

use data/input/GrowthClimateDataset, clear
gen temp = UDel_temp_popweight
reg growthWDI c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id, cluster(iso_id)
	mat b = e(b)
	mat b = b[1,1..2] //save coefficients
	di _b[temp]/-2/_b[c.temp#c.temp]
loc min -5
margins, at(temp=(`min'(1)35)) post noestimcheck level(90)
parmest, norestore level(90)
split parm, p("." "#")
ren parm1 x
destring x, replace
replace x = x + `min' - 1
drop parm*
outsheet using data/output/estimatedGlobalResponse.csv, comma replace  //writing out results for R
use data/input/GrowthClimateDataset, clear
keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year
outsheet using data/output/mainDataset.csv, comma replace
clear
svmat b
outsheet using data/output/estimatedCoefficients.csv, comma replace


// -----------------  DATA FOR FIGURE 2, PANELS B, D, E  -----------------------

loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap
foreach var of loc vars  {
use data/input/GrowthClimateDataset, clear
drop _yi_* _y2_* time time2
gen time = year - 1985
gen time2 = time^2
qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
qui drop _yi_iso_id*
qui drop _y2_iso_id*
gen temp = UDel_temp_popweight
gen poorWDIppp = (GDPpctile_WDIppp<50)
replace poorWDIppp=. if GDPpctile_WDIppp==.
gen interact = poorWDIppp
qui reg `var' interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id) //PPP WDI baseline
loc min 0
margins, over(interact) at(temp=(`min'(1)30)) post noestimcheck force level(90)
parmest, norestore level(90)
split parm, p("." "#")
ren parm1 x
ren parm3 interact
destring x interact, replace
replace x = x - `min' - 1
drop parm*
gen model = "`var'"
if "`var'"=="growthWDI" {
	save data/output/EffectHeterogeneity, replace
	}
	else {
	append using data/output/EffectHeterogeneity
	save data/output/EffectHeterogeneity, replace
	}
}
use data/output/EffectHeterogeneity, clear
outsheet using data/output/EffectHeterogeneity.csv, comma replace


// -----------------  DATA FOR FIGURE 2, PANEL C  -----------------------

use data/input/GrowthClimateDataset, clear
drop _yi_* _y2_* time time2
gen time = year - 1985
gen time2 = time^2
qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
qui drop _yi_iso_id*
qui drop _y2_iso_id*
gen temp = UDel_temp_popweight
gen early = year<1990
gen interact = early
qui reg growthWDI interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
loc min 0
margins, over(interact) at(temp=(`min'(1)30)) post noestimcheck level(90)
parmest, norestore level(90)
split parm, p("." "#")
ren parm1 x
ren parm3 interact
destring x interact, replace
replace x = x - `min' - 1
drop parm*
outsheet using data/output/EffectHeterogeneityOverTime.csv, comma replace
```

---

## PART 2: ORIGINAL R CODE (MakeFigure2.R)

This R code reads the Stata output and creates the figure.

```r
# SCRIPT TO MAKE FIGURE 2
# This file calls data created in GenerateFigure2Data.do

rm(list = ls())

require(maptools)
require(fields)
require(classInt)
require(plotrix)
require(dplyr)
"%&%"<-function(x,y)paste(x,y,sep="")  #define a function for easy string pasting

pdf(file="figures/MainFigs_Input/Figure2.pdf",width=10,height=5.5,useDingbats=F)

mat <- matrix(c(1,1,2,3,1,1,4,5),nrow=2,byrow=T)
layout(mat)
ax = 1.5  #scaling for axes
par(mar=c(4,4,2,1))

########################################################
#  Panel A
########################################################

resp <- read.csv("data/output/estimatedGlobalResponse.csv")
dta <- read.csv("data/output/mainDataset.csv")
smpl <- is.na(dta$growthWDI)==F & is.na(dta$UDel_temp_popweight)==F   #main estimation sample
coef <- read.csv("data/output/estimatedCoefficients.csv")

# center response at optimum
x = resp$x
mx = max(resp$estimate)
est = resp$estimate - mx
min90 = resp$min90 - mx
max90 = resp$max90 - mx

ctys = c('USA','CHN','DEU','JPN','IND','NGA','IDN','BRA','FRA','GBR')
ctt = c('US','CHN',"GER","JPN",'IND','NGA','INDO','BRA','FRA','UK')

#initialize plot
plot(1,xlim=c(-2,30),ylim=c(-0.4,0.1),type="n",las=1,cex.axis=1.3)

# add vertical average temperature lines for selected countries
for (j in 1:length(ctys)) {
  tt = mean(dta$UDel_temp_popweight[dta$iso==ctys[j]],na.rm=T)
  segments(tt,-0.23,tt,0.15,lwd=0.5)
}

# plot CI and main effect
polygon(c(x,rev(x)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(x,est,lwd=2)

# now add histograms at bottom
# first calculate percent of population and global gdp produced at each temperature bin, for our estimation sample
bins = seq(-7,30,0.5)
histtemp = dta$UDel_temp_popweight[smpl]
histpop = dta$Pop[smpl]
histgdp = dta$TotGDP[smpl]
pop = gdp = c()
for (j in 1:(length(bins)-1)) {
  lo = bins[j]
  hi = bins[j+1]
  pop = c(pop,sum(histpop[histtemp>=lo & histtemp<hi]))
  gdp = c(gdp,sum(histgdp[histtemp>=lo & histtemp<hi]))
}
pop = pop/sum(pop)
gdp = gdp/sum(gdp)

#parameters that set where histograms go
dis = 0.055
base = -0.3

# now make histograms
#temperature
zz <- hist(histtemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,col="red")
# pop
cts = pop/max(pop)*0.05
rect(bins,base-dis*(1),bins+0.5,base-dis*(1)+cts,col="grey")
# gdp
cts = gdp/max(gdp)*0.05
rect(bins,base-dis*(2),bins+0.5,base-dis*(2)+cts,col="black")



########################################################
#  Panels b
########################################################
resp <- read.csv("data/output/EffectHeterogeneity.csv")
poor <- dta$GDPpctile_WDIppp<50
rich <- dta$GDPpctile_WDIppp>=50

resp <- resp[resp$x>=5,]  #dropping estimates below 5C, since so little poor country exposure down there
mods = unique(as.character(resp$model))

m <- "growthWDI"
plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
smp = resp$model==m & resp$interact==1  #poor countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
min90 = resp$min90[smp] - mx
max90 = resp$max90[smp] - mx

polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(xx,est,lwd=2,col="steelblue3")

# now add rich countries
smp = resp$model==m & resp$interact==0  #rich countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
lines(xx,est,lwd=2,col="red")

# now add histograms of temperature exposures at the base
bins = seq(-7,30,0.5)
poortemp = dta$UDel_temp_popweight[smpl==T & poor==T]
richtemp = dta$UDel_temp_popweight[smpl==T & rich==T]
base = -0.3
zz <- hist(richtemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,border="red",col="NA")
base = -0.35
zz1 <- hist(poortemp,plot=F,breaks=bins)
cts = zz1$counts/max(zz1$counts)*0.05
rect(bins,base,bins+0.5,base+cts,col="lightblue")


########################################################
#  Panel c
########################################################
resp <- read.csv("data/output/EffectHeterogeneityOverTime.csv")
early <- dta$year<1990

smp = resp$interact==1  #early period
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
min90 = resp$min90[smp] - mx
max90 = resp$max90[smp] - mx

plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(xx,est,lwd=2,col="steelblue3")
# now add point estimate for later period
smp = resp$interact==0  #poor countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
lines(xx,est,lwd=2,col="red")

# now add histograms of temperature exposures at the base
bins = seq(-7,30,0.5)
earlytemp = dta$UDel_temp_popweight[smpl==T & early==T]
latetemp = dta$UDel_temp_popweight[smpl==T & early==F]
base = -0.3
zz <- hist(earlytemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,border="red",col=NA)
base = -0.35
zz1 <- hist(latetemp,plot=F,breaks=bins)
cts = zz1$counts/max(zz1$counts)*0.05
rect(bins,base,bins+0.5,base+cts,col="lightblue")


########################################################
#  Panels d, e
########################################################

resp <- read.csv("data/output/EffectHeterogeneity.csv")
poor <- dta$GDPpctile_WDIppp<50
rich <- dta$GDPpctile_WDIppp>=50
resp <- resp[resp$x>=5,]  #dropping estimates below 5C, because so little poor country exposure there
mods = unique(as.character(resp$model))
toplot=c("AgrGDPgrowthCap","NonAgrGDPgrowthCap")

for (m in toplot) {
  plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
  smp = resp$model==m & resp$interact==1  #poor countries
  xx = resp$x[smp]
  mx = max(resp$estimate[smp])
  est = resp$estimate[smp] - mx
  min90 = resp$min90[smp] - mx
  max90 = resp$max90[smp] - mx

  polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
  lines(xx,est,lwd=2,col="steelblue3")
  # now add rich countries
  smp = resp$model==m & resp$interact==0  #poor countries
  xx = resp$x[smp]
  mx = max(resp$estimate[smp])
  est = resp$estimate[smp] - mx
  lines(xx,est,lwd=2,col="red")

}

dev.off()
```

---

## PART 3: PYTHON TRANSLATION

### Panel A: generate_global_response()

```python
def generate_global_response(self, results):
    """
    Generate the global temperature-growth response curve ("damage function").

    PURPOSE:
    Create a curve showing how GDP growth varies with temperature across the full range
    of observed temperatures (-5C to 35C annual average).

    WHAT THIS FUNCTION DOES:
    1. Calculate optimal temperature (where growth is maximized)
    2. For each temperature from -5C to 35C:
       - Predict growth rate using: B1*T + B2*T^2
       - Calculate 90% confidence interval
    3. Normalize by subtracting maximum (like R code does)
    4. Save the curve for plotting (this becomes Figure 2, Panel A)

    MATCHING STATA/R APPROACH:
    1. Stata: margins, at(temp=(-5(1)35)) post noestimcheck level(90)
       - Computes predictions and CIs at each temperature
    2. R: mx = max(resp$estimate); est = resp$estimate - mx; min90 = resp$min90 - mx
       - Normalizes by subtracting a CONSTANT (the max estimate) from everything
       - This preserves CI width at each temperature point

    Original Stata code:
    margins, at(temp=(-5(1)35)) post noestimcheck level(90)
    parmest, norestore level(90)
    """
    logger.info("Generating global response function...")

    # Get coefficients
    temp_coef = results.params['UDel_temp_popweight']
    temp2_coef = results.params['UDel_temp_popweight_2']

    # Calculate optimal temperature (where growth is maximized)
    optimal_temp = -temp_coef / (2 * temp2_coef)

    # Generate temperature range for response function (like Stata: margins, at(temp=(-5(1)35)))
    temp_range = np.arange(-5, 36, 1)

    # Step 1: Calculate predicted growth rates (raw, not normalized yet)
    # This is: y(T) = B1*T + B2*T^2
    predictions_raw = temp_coef * temp_range + temp2_coef * temp_range ** 2

    # Step 2: Get variance-covariance matrix for temperature coefficients AND constant
    # Stata's margins includes variance from the constant term, giving non-zero SE at T=0
    temp_vars = ['UDel_temp_popweight', 'UDel_temp_popweight_2']
    all_vars = ['const'] + temp_vars
    full_cov = results.cov_params().loc[all_vars, all_vars].values

    # Step 3: Calculate standard errors for RAW predictions (before normalization)
    # Using gradient [1, T, T^2] for y(T) = const + B1*T + B2*T^2
    # Including the constant gives non-zero SE at T=0, matching Stata's margins
    se_predictions = []
    for temp in temp_range:
        # Gradient includes constant term: [d/d_const, d/d_B1, d/d_B2] = [1, T, T^2]
        grad = np.array([1, temp, temp**2])
        pred_var = grad.T @ full_cov @ grad
        se_predictions.append(np.sqrt(max(0, pred_var)))  # Ensure non-negative

    se_predictions = np.array(se_predictions)

    # Step 4: Calculate confidence intervals for raw predictions (90% CI like Stata)
    ci_factor = stats.norm.ppf(0.95)  # 90% CI
    lower_ci_raw = predictions_raw - ci_factor * se_predictions
    upper_ci_raw = predictions_raw + ci_factor * se_predictions

    # Step 5: Normalize by subtracting the maximum estimate (matching R code exactly)
    # R code: mx = max(resp$estimate); est = resp$estimate - mx
    #         min90 = resp$min90 - mx; max90 = resp$max90 - mx
    mx = np.max(predictions_raw)
    predictions = predictions_raw - mx
    lower_ci = lower_ci_raw - mx
    upper_ci = upper_ci_raw - mx

    # Create response function dataframe
    response_data = pd.DataFrame({
        'x': temp_range,
        'estimate': predictions,
        'min90': lower_ci,
        'max90': upper_ci
    })

    # Save results
    response_data.to_csv(OUTPUT_FILES['estimated_global_response'], index=False)

    # Save coefficients (like Stata: mat b = e(b); mat b = b[1,1..2])
    coef_data = pd.DataFrame({
        'temp': [temp_coef],
        'temp2': [temp2_coef]
    })
    coef_data.to_csv(OUTPUT_FILES['estimated_coefficients'], index=False)

    logger.info(f"Global response function saved. Optimal temperature: {optimal_temp:.2f}C")
    return response_data
```

### Panels B, D, E: heterogeneity_analysis()

```python
def heterogeneity_analysis(self):
    """
    Analyze heterogeneity in temperature responses (Figure 2, panels B, D, E).

    Original Stata code:
    loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap
    foreach var of loc vars  {
    ...
    qui reg `var' interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
    margins, over(interact) at(temp=(0(1)30)) post noestimcheck force level(90)
    ...
    """
    logger.info("Running heterogeneity analysis...")

    results_list = []

    # Variables to analyze (like Stata: loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap)
    variables = ['growthWDI', 'AgrGDPgrowthCap', 'NonAgrGDPgrowthCap']

    for var in variables:
        if var not in self.data.columns:
            logger.warning(f"Variable {var} not found in dataset, skipping...")
            continue

        logger.info(f"Analyzing heterogeneity for {var}...")

        # Use unified regression function for heterogeneity analysis
        result_dict = self.run_regression('heterogeneity',
                                       dependent_var=var,
                                       interaction_var='poorWDIppp')

        results = result_dict['results']

        # Get variance-covariance matrix for all relevant coefficients
        # Include const, temp, temp2, temp_poor, temp2_poor
        cov_vars = ['const', 'UDel_temp_popweight', 'UDel_temp_popweight_2',
                   'temp_poor', 'temp2_poor']
        full_cov = results.cov_params().loc[cov_vars, cov_vars].values

        # Generate response functions for rich and poor (like Stata: margins, over(interact) at(temp=(0(1)30)))
        temp_range = np.arange(0, 31, 1)
        ci_factor = stats.norm.ppf(0.95)  # 90% CI

        for interact in [0, 1]:  # 0 = rich, 1 = poor
            if interact == 0:
                # Rich countries (no interaction)
                temp_coef = results.params['UDel_temp_popweight']
                temp2_coef = results.params['UDel_temp_popweight_2']
            else:
                # Poor countries (with interaction)
                temp_coef = results.params['UDel_temp_popweight'] + results.params['temp_poor']
                temp2_coef = results.params['UDel_temp_popweight_2'] + results.params['temp2_poor']

            # Calculate raw predictions
            predictions_raw = temp_coef * temp_range + temp2_coef * temp_range ** 2

            # Calculate standard errors using delta method
            # Gradient for [const, temp, temp2, temp_poor, temp2_poor]:
            # - Rich: [1, T, T^2, 0, 0]
            # - Poor: [1, T, T^2, T, T^2]
            se_predictions = []
            for temp in temp_range:
                if interact == 0:
                    grad = np.array([1, temp, temp**2, 0, 0])
                else:
                    grad = np.array([1, temp, temp**2, temp, temp**2])
                pred_var = grad.T @ full_cov @ grad
                se_predictions.append(np.sqrt(max(0, pred_var)))

            se_predictions = np.array(se_predictions)

            # Calculate raw CIs
            lower_ci_raw = predictions_raw - ci_factor * se_predictions
            upper_ci_raw = predictions_raw + ci_factor * se_predictions

            # Normalize by subtracting the maximum estimate (matching R code)
            mx = np.max(predictions_raw)
            predictions = predictions_raw - mx
            lower_ci = lower_ci_raw - mx
            upper_ci = upper_ci_raw - mx

            # Create result rows
            for i, temp_val in enumerate(temp_range):
                results_list.append({
                    'x': temp_val,
                    'estimate': predictions[i],
                    'min90': lower_ci[i],
                    'max90': upper_ci[i],
                    'interact': interact,
                    'model': var
                })

    # Save heterogeneity results
    heterogeneity_data = pd.DataFrame(results_list)
    heterogeneity_data.to_csv(OUTPUT_FILES['effect_heterogeneity'], index=False)

    logger.info("Heterogeneity analysis completed")
    return heterogeneity_data
```

### Panel C: temporal_heterogeneity()

```python
def temporal_heterogeneity(self):
    """
    Analyze temporal heterogeneity (Figure 2, panel C).

    Original Stata code:
    gen early = year<1990
    gen interact = early
    qui reg growthWDI interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
    margins, over(interact) at(temp=(0(1)30)) post noestimcheck level(90)
    """
    logger.info("Running temporal heterogeneity analysis...")

    # Use unified regression function for temporal heterogeneity analysis
    result_dict = self.run_regression('temporal',
                                   dependent_var='growthWDI',
                                   interaction_var='early')

    results = result_dict['results']

    # Get variance-covariance matrix for all relevant coefficients
    # Include const, temp, temp2, temp_early, temp2_early
    cov_vars = ['const', 'UDel_temp_popweight', 'UDel_temp_popweight_2',
               'temp_early', 'temp2_early']
    full_cov = results.cov_params().loc[cov_vars, cov_vars].values

    # Generate response functions for early and late periods
    temp_range = np.arange(0, 31, 1)
    ci_factor = stats.norm.ppf(0.95)  # 90% CI
    results_list = []

    for interact in [0, 1]:  # 0 = late, 1 = early
        if interact == 0:
            # Late period (no interaction)
            temp_coef = results.params['UDel_temp_popweight']
            temp2_coef = results.params['UDel_temp_popweight_2']
        else:
            # Early period (with interaction)
            temp_coef = results.params['UDel_temp_popweight'] + results.params['temp_early']
            temp2_coef = results.params['UDel_temp_popweight_2'] + results.params['temp2_early']

        # Calculate raw predictions
        predictions_raw = temp_coef * temp_range + temp2_coef * temp_range ** 2

        # Calculate standard errors using delta method
        # Gradient for [const, temp, temp2, temp_early, temp2_early]:
        # - Late: [1, T, T^2, 0, 0]
        # - Early: [1, T, T^2, T, T^2]
        se_predictions = []
        for temp in temp_range:
            if interact == 0:
                grad = np.array([1, temp, temp**2, 0, 0])
            else:
                grad = np.array([1, temp, temp**2, temp, temp**2])
            pred_var = grad.T @ full_cov @ grad
            se_predictions.append(np.sqrt(max(0, pred_var)))

        se_predictions = np.array(se_predictions)

        # Calculate raw CIs
        lower_ci_raw = predictions_raw - ci_factor * se_predictions
        upper_ci_raw = predictions_raw + ci_factor * se_predictions

        # Normalize by subtracting the maximum estimate (matching R code)
        mx = np.max(predictions_raw)
        predictions = predictions_raw - mx
        lower_ci = lower_ci_raw - mx
        upper_ci = upper_ci_raw - mx

        # Create result rows
        for i, temp_val in enumerate(temp_range):
            results_list.append({
                'x': temp_val,
                'estimate': predictions[i],
                'min90': lower_ci[i],
                'max90': upper_ci[i],
                'interact': interact
            })

    # Save temporal heterogeneity results
    temporal_data = pd.DataFrame(results_list)
    temporal_data.to_csv(OUTPUT_FILES['effect_heterogeneity_time'], index=False)

    logger.info("Temporal heterogeneity analysis completed")
    return temporal_data
```

---

## KEY QUESTIONS FOR ANALYSIS

1. **Stata's `margins` command**: What exactly does `margins, at(temp=(-5(1)35)) post noestimcheck level(90)` compute? Does it compute predictions for just the temperature terms, or for the full model? How does it compute standard errors?

2. **The `parmest` output**: What columns does `parmest` produce? Does it include `estimate`, `min90`, `max90`? Are these marginal effects or predicted values?

3. **Python's approach**: The Python code computes:
   - Predictions as `B1*T + B2*T^2` (just temperature terms)
   - Standard errors using delta method with gradient `[1, T, T^2]` including the constant
   - Then normalizes by subtracting max(predictions)

   Is this equivalent to what Stata's margins produces?

4. **Potential discrepancy**: Stata's `margins` might be computing predictions that include all model terms (averaged over fixed effects), while Python is computing just the temperature polynomial. This could lead to different standard errors.

5. **Clustered standard errors**: The Stata regression uses `cluster(iso_id)`. Does this affect how `margins` computes standard errors? Is Python's variance-covariance matrix from statsmodels equivalent?
