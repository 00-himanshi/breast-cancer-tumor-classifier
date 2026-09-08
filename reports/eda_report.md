# Advanced Exploratory Data Analysis Report

## 1. Basic Information
- **Total Samples:** 569
- **Total Features:** 32
- **Detected Target Column:** `diagnosis`

## 2. Missing Values
No missing values found in the dataset.

## 3. Near-Zero Variance Analysis
Features with zero or near-zero variance carry little to no information.
```text
Features with variance < 1e-4:
fractal_dimension_mean    0.000050
smoothness_se             0.000009
concave points_se         0.000038
symmetry_se               0.000068
fractal_dimension_se      0.000007
dtype: float64
```

## 4. IQR-Based Outlier Analysis
Outliers detected using the Interquartile Range (IQR) method (1.5 * IQR).
```text
                         Outlier Count  Percentage (%)
area_se                             65       11.423550
radius_se                           38        6.678383
perimeter_se                        38        6.678383
area_worst                          35        6.151142
smoothness_se                       30        5.272408
compactness_se                      28        4.920914
fractal_dimension_se                28        4.920914
symmetry_se                         27        4.745167
area_mean                           25        4.393673
fractal_dimension_worst             24        4.217926
symmetry_worst                      23        4.042179
concavity_se                        22        3.866432
texture_se                          20        3.514938
concave points_se                   19        3.339192
concavity_mean                      18        3.163445
radius_worst                        17        2.987698
compactness_worst                   16        2.811951
compactness_mean                    16        2.811951
perimeter_worst                     15        2.636204
symmetry_mean                       15        2.636204
fractal_dimension_mean              15        2.636204
radius_mean                         14        2.460457
perimeter_mean                      13        2.284710
concavity_worst                     12        2.108963
concave points_mean                 10        1.757469
smoothness_worst                     7        1.230228
texture_mean                         7        1.230228
smoothness_mean                      6        1.054482
texture_worst                        5        0.878735
```

## 5. Target Class Distribution
```text
diagnosis
B    357
M    212
Name: count, dtype: int64

diagnosis
B    62.741652
M    37.258348
Name: proportion, dtype: float64 %
```

![Target Distribution](target_distribution.png)

## 6. Target Correlation Analysis
Correlation of numeric features with the encoded target variable:
```text
concave points_worst       0.793566
perimeter_worst            0.782914
concave points_mean        0.776614
radius_worst               0.776454
perimeter_mean             0.742636
area_worst                 0.733825
radius_mean                0.730029
area_mean                  0.708984
concavity_mean             0.696360
concavity_worst            0.659610
compactness_mean           0.596534
compactness_worst          0.590998
radius_se                  0.567134
perimeter_se               0.556141
area_se                    0.548236
texture_worst              0.456903
smoothness_worst           0.421465
symmetry_worst             0.416294
texture_mean               0.415185
concave points_se          0.408042
smoothness_mean            0.358560
symmetry_mean              0.330499
fractal_dimension_worst    0.323872
compactness_se             0.292999
concavity_se               0.253730
fractal_dimension_se       0.077972
smoothness_se             -0.067016
fractal_dimension_mean    -0.012838
texture_se                -0.008303
symmetry_se               -0.006522
dtype: float64
```

![Correlation Heatmap](correlation_heatmap.png)

## 7. Class-wise Feature Distributions
Visualizing how the distributions of the top 12 most correlated features differ between classes.

![Class-wise Feature Distributions](classwise_distributions.png)

