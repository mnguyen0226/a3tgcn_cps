# MAHALANOBIS DISTANCE CALCULATION
import numpy as np
from pyod.models.copod import COPOD

#Load libraries
import copy 
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

# Collect 5 predictions and labesl
pred_arr1 = np.array(
    [
        0.020174682,
        0.012799442,
        0.04910487,
        0.03856057,
        -0.007079959,
        0.045596898,
        0.017695367,
        0.78167224,
        0.7632853,
        -0.0014361143,
        0.03818363,
        0.0034386516,
        0.027982235,
        0.41673723,
        0.29678136,
        -0.0031874776,
        0.21287149,
        0.0034342408,
        0.5172912,
        0.040563583,
        0.29257208,
        0.22969091,
        0.5992017,
        0.22986543,
        0.70996606,
        0.17503798,
        0.667894,
        0.20405924,
        0.5570904,
        0.27468914,
        0.2492252,
    ]
)
gt_arr1 = np.array(
    [
        0.014630047,
        0.041327894,
        0.03186278,
        0.02798142,
        0.025564658,
        0.043613024,
        0.033620954,
        0.7889404,
        0.78909576,
        0.0,
        0.0,
        0.0,
        0.0,
        0.41012758,
        0.3001206,
        0.0,
        0.0,
        0.0,
        0.55948144,
        0.024245884,
        0.28269085,
        0.24582922,
        0.5931442,
        0.24549505,
        0.7075933,
        0.2132407,
        0.69135606,
        0.2132407,
        0.4756116,
        0.2659462,
        0.25861156,
    ]
)
pred_arr2 = np.array(
    [
        0.022113264,
        0.016278684,
        0.05012375,
        0.03939259,
        -0.0008433461,
        0.051617563,
        0.029655814,
        0.7764827,
        0.76057327,
        0.00063312054,
        0.10921562,
        0.006396711,
        0.020545721,
        0.3687722,
        0.2959867,
        0.0010075569,
        0.16619748,
        -0.0018725991,
        0.49081168,
        0.0357123,
        0.27906996,
        0.22922522,
        0.6196858,
        0.22639072,
        0.68368113,
        0.1864807,
        0.6573744,
        0.20318091,
        0.543315,
        0.27003348,
        0.23229975,
    ]
)
gt_arr2 = np.array(
    [
        0.015325891,
        0.042892326,
        0.0291781,
        0.030894324,
        0.02962966,
        0.040735655,
        0.020923415,
        0.7918894,
        0.7920456,
        0.0,
        0.0,
        0.0,
        0.0,
        0.41249165,
        0.29964167,
        0.0,
        0.0,
        0.0,
        0.54006565,
        0.024244478,
        0.28110564,
        0.24679776,
        0.5786184,
        0.24646,
        0.70387447,
        0.21260789,
        0.69159776,
        0.21260789,
        0.44282165,
        0.2668325,
        0.25949785,
    ]
)
pred_arr3 = np.array(
    [
        0.019025564,
        0.019257963,
        0.020491064,
        0.037564695,
        0.0097644925,
        0.056824625,
        0.03129983,
        0.7711693,
        0.77298284,
        0.006313026,
        0.18510228,
        -0.0032339096,
        0.015606821,
        0.39016703,
        0.2745064,
        0.0018710494,
        0.22068691,
        0.0060781837,
        0.49073875,
        0.033984244,
        0.27227736,
        0.21328366,
        0.6400367,
        0.22226018,
        0.6779504,
        0.19517034,
        0.6454435,
        0.21178228,
        0.54523957,
        0.2709626,
        0.2361505,
    ]
)
gt_arr3 = np.array(
    [
        0.014836625,
        0.044092678,
        0.026007602,
        0.032453965,
        0.032211695,
        0.041638777,
        0.019598806,
        0.79178184,
        0.79193807,
        0.0,
        0.0,
        0.0,
        0.0,
        0.41439322,
        0.0,
        0.0,
        0.26587144,
        0.0,
        0.53193724,
        0.024244528,
        0.28116345,
        0.24774434,
        0.5595979,
        0.24740368,
        0.7010178,
        0.21697892,
        0.50851226,
        0.21596514,
        0.5504633,
        0.2677519,
        0.26041725,
    ]
)
pred_arr4 = np.array(
    [
        -0.003928423,
        0.022525132,
        0.028811753,
        0.033315063,
        0.015749872,
        0.052277982,
        0.027225554,
        0.76673627,
        0.7764274,
        0.01301533,
        0.22202826,
        -0.0023428798,
        0.008938491,
        0.38484085,
        0.115195334,
        0.010332525,
        0.30119473,
        0.011395633,
        0.5407618,
        0.035272956,
        0.26958996,
        0.1955939,
        0.6582716,
        0.21987176,
        0.6829926,
        0.19732112,
        0.58377916,
        0.21664262,
        0.55171263,
        0.2733873,
        0.24566585,
    ]
)
gt_arr4 = np.array(
    [
        0.015368411,
        0.043890193,
        0.025964025,
        0.03284907,
        0.025489133,
        0.04293711,
        0.019486684,
        0.79049134,
        0.79064715,
        0.0,
        0.29355296,
        0.0,
        0.0,
        0.41083482,
        0.0,
        0.0,
        0.2652303,
        0.0,
        0.5700275,
        0.024245145,
        0.28185743,
        0.2406749,
        0.7322637,
        0.24125351,
        0.70195496,
        0.21787249,
        0.5045648,
        0.21686323,
        0.55367553,
        0.26439223,
        0.25705758,
    ]
)
pred_arr5 = np.array(
    [
        -0.00025236607,
        0.023543,
        0.047958374,
        0.039734542,
        0.0197528,
        0.045108855,
        0.02223885,
        0.77052146,
        0.77818066,
        0.011674941,
        0.29735428,
        -0.008002341,
        0.012838304,
        0.3964926,
        0.08415711,
        0.011705995,
        0.24603891,
        0.0071329474,
        0.55817163,
        0.03707534,
        0.27438527,
        0.20667064,
        0.72421527,
        0.22117704,
        0.6772195,
        0.22711366,
        0.5591305,
        0.20631516,
        0.5498364,
        0.26401472,
        0.2440598,
    ]
)
gt_arr5 = np.array(
    [
        0.01594372,
        0.042522274,
        0.030146204,
        0.03380061,
        0.019144015,
        0.04444103,
        0.020412292,
        0.7918307,
        0.79198694,
        0.0,
        0.2979866,
        0.0,
        0.0,
        0.40912578,
        0.0,
        0.0,
        0.26292917,
        0.0,
        0.57678926,
        0.024244506,
        0.2811372,
        0.23935473,
        0.7230194,
        0.23996165,
        0.7040325,
        0.21808398,
        0.50293356,
        0.21709089,
        0.5621446,
        0.2631632,
        0.2558286,
    ]
)

pred_arr6 = np.array([0.01637503,0.041198634,0.03391889,0.035160873,0.013488324,0.04482272,0.025299462,0.79170734,0.79186344,0.0,0.29013368,0.0,0.0,0.41290292,0.0,0.0,0.25932944,0.0,0.58699596,0.024244566,0.28120357,0.23846506,0.7360706,0.2390209,0.695615,0.21892956,0.49275237,0.21796149,0.5757046,0.26223797,0.25490332])
pred_arr7 = np.array([0.015848065,0.039890174,0.03800128,0.033599105,0.014216439,0.04482272,0.026946435,0.7897942,0.7899498,0.0,0.29015124,0.0,0.0,0.40367985,0.28527513,0.0,0.24487936,0.0,0.59372705,0.024245476,0.28223214,0.23760091,0.7351756,0.23817079,0.7129241,0.16680364,0.67177695,0.16593309,0.57215244,0.2613101,0.2539755])
pred_arr8 = np.array([0.0154407695,0.039263003,0.04195396,0.035550803,0.018742446,0.04482272,0.029777259,0.788691,0.78884625,0.0,0.28886288,0.0,0.0,0.0,0.2849898,0.0,0.24117786,0.0,0.54122996,0.024246003,0.2828248,0.24736547,0.7471866,0.24825248,0.5336354,0.1677452,0.6731959,0.16689888,0.58490986,0.26754728,0.26021266])
pred_arr9 = np.array([0.015493573,0.042908918,0.042635452,0.024906218,0.022975909,0.04482272,0.034556497,0.7826086,0.78276205,0.0,0.0,0.0,0.0,0.0,0.28657898,0.0,0.24221529,0.0,0.43501556,0.024248889,0.28608733,0.2632274,0.6004079,0.2632274,0.5041139,0.16890319,0.6716119,0.1680501,0.5827815,0.2812369,0.2739023])
pred_arr10 = np.array([0.01550364,0.044441488,0.03979943,0.02585827,0.026416337,0.04482272,0.038110632,0.7891227,0.7892781,0.0,0.0,0.0,0.0,0.4127917,0.2851193,0.0,0.2416207,0.0,0.5336811,0.024245797,0.28259292,0.24818264,0.59172356,0.24784443,0.70466197,0.1674414,0.6726707,0.1665922,0.58320546,0.26818767,0.26085305])
gt_arr6 = np.array([0.0063304305,0.032058537,0.03727609,0.04068452,0.015947104,0.046176553,0.023760736,0.7694875,0.7808867,0.008980751,0.29757392,-0.0035626888,0.008166075,0.38652867,0.10584211,0.011305988,0.22491199,0.004882753,0.5550586,0.043658018,0.2810455,0.20479763,0.723145,0.22813028,0.680002,0.22615016,0.56466377,0.2005102,0.5458876,0.2600239,0.25882035])
gt_arr7 = np.array([0.015928686,0.04050678,0.03767556,0.036990166,0.020227969,0.05057335,0.022577882,0.7671836,0.79152596,0.00258255,0.29205155,-0.0043300986,0.00610584,0.35648912,0.18941718,0.010856152,0.22465926,0.00071805716,0.556819,0.054507494,0.29055542,0.22269768,0.7416518,0.23649079,0.68083346,0.2166273,0.62246364,0.20009619,0.54021746,0.26550186,0.2619818])
gt_arr8 = np.array([0.027346134,0.032170594,0.046040475,0.043251038,0.018333912,0.04851222,0.020858169,0.76969,0.7977576,-0.003712356,0.20787317,-0.012580752,0.016517699,0.34295893,0.31516546,4.6789646e-05,0.21916121,0.010444224,0.53524363,0.059065998,0.29065198,0.23692173,0.73513055,0.2410841,0.67344904,0.19551057,0.68542486,0.20510334,0.5427711,0.2681644,0.2509033])
gt_arr9 = np.array([0.017722547,0.025940359,0.05041504,0.040984154,0.011931717,0.050291717,0.021306872,0.77594894,0.7964695,-0.0075377226,0.053223073,-0.020846903,0.011800349,0.21622795,0.30108732,-0.0068289638,0.2247479,0.013486326,0.5072416,0.053462327,0.28866285,0.25916308,0.6738974,0.24279231,0.65343285,0.17208087,0.671631,0.21062511,0.5435599,0.27784294,0.23705667])
gt_arr10 = np.array([0.023599863,0.018085659,0.058042467,0.038346708,0.024487853,0.05321461,0.029597402,0.77984077,0.7739598,-0.0075009465,-0.018564165,-0.014595926,0.023602307,0.30325204,0.25586098,-0.0037098527,0.21007591,0.013257563,0.42615926,0.039732397,0.2851941,0.26403546,0.60098076,0.24477726,0.6792157,0.18614751,0.6627444,0.21371692,0.54371065,0.27617568,0.23415262])

# Calculates errors
error_1 = gt_arr1 - pred_arr1
error_2 = gt_arr2 - pred_arr2
error_3 = gt_arr3 - pred_arr3
error_4 = gt_arr4 - pred_arr4
error_5 = gt_arr5 - pred_arr5
error_6 = gt_arr6 - pred_arr6
error_7 = gt_arr7 - pred_arr7
error_8 = gt_arr8 - pred_arr8
error_9 = gt_arr9 - pred_arr9
error_10 = gt_arr10 - pred_arr10

# Stacks errors - create a database
error_stack = np.stack((error_1, error_2, error_3, error_4, error_5, error_6, error_7, error_8, error_9, error_10))

########################
#Mahalonibis Distance

def robust_mahalanobis_method(df):
    #Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df, rowvar=False)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = sp.linalg.inv(mcd) #inverse covariance metric
    
    #Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    #Flag as outlier
    outlier = []
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

outliers_mahal_rob_bi, md_rb_bi = robust_mahalanobis_method(df=error_stack)
#[141, 374, 380, 398, 404, 405, 410, 414, 418, 427]

outliers_mahal_rob, md_rb = robust_mahalanobis_method(df=error_stack)
#[123, 126, 142, 152, 155, 163, 214, 283, 353, 364, 365, 367, 380, 405, 410, 
# 418, 488, 489, 490, 491, 492]