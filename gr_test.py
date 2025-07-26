import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 
import datetime as dt


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42




def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    ----------
    x : array_like
        List of floats representing the x values of the data (MUST BE ORDERED)
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


cdf = pd.read_csv('data/PointLoma_sewage_qPCR.csv')
cdf['Sample_Date'] = pd.to_datetime(cdf['Sample_Date'])
cdf =cdf.set_index('Sample_Date')
import numpy as np

## interpolate qPCR data

numberDates = [dvi.value/10**11 for dvi in cdf.index]

cdf['Mean viral gene copies/L'] = non_uniform_savgol(numberDates,cdf['Mean viral gene copies/L'],5,1)
# convert to daily via interpolation
date_ind = pd.date_range(cdf.index.min(),cdf.index.max(),freq='D')
cdf = cdf.reindex(date_ind)
cdf = cdf.interpolate()

fig,ax = plt.subplots()
ax.plot(cdf.index,cdf['Mean viral gene copies/L'])
ax.set_yscale('log')

locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax.set_xlim([cdf.index.min(),cdf.index.max()])
plt.setp(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Mean viral gene copies/L')
fig.tight_layout()
plt.savefig('smoothed_viral_loads.pdf')


#### now, solve the deconvolution problem to get a proxy for cases per day

# alpha = 0.3
# beta = alpha/0.624

# A = 1
# C0 = A*alpha/(beta-alpha)
t = np.linspace(0,40,41)
# c = C0*np.exp(-alpha*t)*(1-np.exp(-(beta-alpha)*t))
# ### estimate growth rates
# ## shedding kernel
def eclipse_model(y, t, b, k, delta, p, mu, c):
    T, I1, I2, Vi, Vni = y
    dydt = [-b*Vi*T, b*Vi*T - k*I1, k*I1- delta*I2, p*mu*I2 - c*Vi, p*(1.-mu)*I2 - c*Vni]
    return dydt

b = 5e-5#1.5e-5
c =10
k= 6#4
mu = 1.0e-4
p = 1e5#1.1e6
delta = 0.5#0.33
from scipy.integrate import odeint
y0 = [1.33e5, 0, 1./30, 0, 0]

sol = odeint(eclipse_model, y0, t, args=(b, k, delta, p, mu, c))

c = sol[:,3:].sum(axis=1)
#simulate simple 
fig,ax = plt.subplots()
ax.plot(np.arange(0,len(t),1),c)
ax.set_yscale('log')
# ax.set_ylim([0,1000])
fig.tight_layout()
plt.savefig('shedding_curve_log.pdf')
plt.close('all')

c = c/c.sum()
#simulate simple 
fig,ax = plt.subplots()
ax.plot(np.arange(0,len(t),1),c)
ax.set_xlabel('Days since infection')
ax.set_ylabel('Shedding Load Density')
# ax.set_yscale('log')
# ax.set_ylim([0,1000])
fig.tight_layout()
plt.savefig('shedding_curve_normalized.pdf')
plt.close('all')

# ##

shed_df = pd.read_csv('inferred_kernel.csv',header=0,index_col='lags')['load']
shed_df = shed_df.apply(lambda x: 0 if x<0 else x)

# c = np.array(shed_df.iloc[1:].values)

c = c[1:] #drops first day, concentration=0
# c = c/np.sum(c)
cFlipped = c[::-1]
F = len(c)


# from scipy.signal import deconvolve
# recovered, remainder = deconvolve(cdf['Mean viral gene copies/L'], c)
# ### now convert ww trajectory into infections

tiledMat = np.zeros((len(cdf),len(cdf)))
for j in range(0,len(cdf)):
    tiledMat[j,np.max([j-len(c)+1,0]):(j+1)] = cFlipped[np.max([len(c)-j-1,0]):len(c)]

# tiledMat = tiledMat[F:(len(c)-F,:]
infects = np.dot(np.linalg.pinv(tiledMat),cdf)
infects = pd.Series(infects[:,0],index=cdf.index)

cases = pd.read_csv('data/cases.csv',index_col=0)
cases = cases[cases['catchment'] =='PointLoma']
cases = cases[(cases.index>='2021-04-01') & (cases.index<='2022-07-01')]
cases.index = pd.to_datetime(cases.index)
cases = cases.groupby(pd.Grouper(freq='D'))[['new_cases']].sum()
cases['cases'] = cases['new_cases'].rolling(window=7, center=True, min_periods=0).mean()


minInd = np.max([cdf.index.min(),cases.index.min()])
maxInd = np.min([cdf.index.max(),cases.index.max()])

cdf = cdf[cdf.index>=minInd]
cdf = cdf[cdf.index<=maxInd]

cases = cases[cases.index>=minInd]
cases = cases[cases.index<=maxInd]

infects = infects[infects.index>=minInd]
infects = infects[infects.index<=maxInd]



import cvxpy as cp
# least squares problem
x = cp.Variable(1)
cost = cp.norm(cp.multiply(infects,x) - cases['cases'],2)
## add some constraints on the shape of the curve. 
# constraints = [x >= 0]
prob = cp.Problem(cp.Minimize(cost))#,constraints)
prob.solve(verbose=True,solver=cp.CLARABEL)


infects = infects*x.value
infects = infects.astype(int)
# infects = infects[infects>0]
fig,ax = plt.subplots()
ax.plot(cdf.index,cases['cases'],color='black')
ax2 = ax.twinx()
# ax2.plot(cdf.index,cdf['Mean viral gene copies/L'])
ax2.plot(cdf.index,infects)
fig.tight_layout()
plt.savefig('infections_curve.pdf')
plt.close('all')

infects = infects[infects>0]

import epyestim
import epyestim.covid19 as covid19

date_ind = pd.date_range(infects.index.min(),infects.index.max(),freq='D')
infects = infects.reindex(date_ind)
infects = infects.fillna(0)
# infects0 = pd.Series(infects[5:len(infects)-5,0],index=cdf.index[5:len(infects)-5])
ww_time_varying_r = covid19.r_covid(infects)
clin_time_varying_r = covid19.r_covid(cases['cases'])


fig,ax = plt.subplots()
ax.plot(ww_time_varying_r.index,ww_time_varying_r.loc[:,'Q0.5'],color='cornflowerblue')
ax.fill_between(ww_time_varying_r.index, ww_time_varying_r['Q0.025'], ww_time_varying_r['Q0.975'], color='cornflowerblue', alpha=0.2)
ax.plot(clin_time_varying_r.index,clin_time_varying_r.loc[:,'Q0.5'],color='black')
ax.fill_between(clin_time_varying_r.index, clin_time_varying_r['Q0.025'], clin_time_varying_r['Q0.975'], color='black', alpha=0.2)

plt.savefig('ww_Re_estimate.pdf')
