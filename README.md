# Estimating Effective Reproduction Number from Wastewater Surveillance

**Elizabeth Murphy** — SURI Internship, Scripps Research Translational Institute  
Advisors: Joshua Levy PhD, Kristian Andersen PhD

## Overview
Wastewater surveillance captures SARS-CoV-2 signal from the entire community — including asymptomatic and untested individuals — making it a more complete indicator of infection dynamics than clinical case counts alone. This project builds a pipeline to estimate the effective reproduction number (Re) directly from wastewater viral load data, using San Diego County's Point Loma treatment plant as a case study (April 2021 – July 2022).

The approach learns a viral shedding kernel from paired wastewater and clinical data, deconvolves community-level infections from the wastewater signal, and estimates Rₑ via the EpiEstim renewal equation. Wastewater-derived Rₑ estimates closely tracked clinically-derived estimates across all major transmission waves, and remained informative even as clinical testing declined in mid-2022.

## Methods
1. Smooth wastewater signal — non-uniform Savitzky-Golay filter on irregularly-spaced qPCR measurements, interpolated to a daily grid
2. Learn shedding kernel — constrained least-squares (CVXPY) regression of wastewater viral load on lagged clinical cases, with shape constraints enforcing a biologically plausible rising and declining profile
3. Deconvolve infections — pseudoinverse of the convolution matrix recovers daily infection counts from wastewater; scaled to match clinical case magnitude
4. Estimate Re — applied to both wastewater-inferred and reported case counts using epyestim, with 95% credible intervals
