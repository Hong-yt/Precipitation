# The code for manuscript: Predicting precipitation satellite sensor accuracy in ungauged areas using explainable machine learning.
The multiplicative triple collocation Analysis (MTC) method is a data evaluation technique that estimates the accuracy of three datasets by pairwise comparison of three independent observation samples. 
The extended double instrumental variable algorithm（EIVD） method only requires two remote sensing datasets with error-related correlations and a tool variable that is strongly correlated with the true value but unrelated to the data errors.
Explainable machine learning model（DNN） for accuracy prediction of satellite remote sensing precipitation data in ungauged areas.

## 1. Environments setting

We recommend using a GPU environment and please first confirm that the cuda version is 11.3. Then, please Install the required dependencies “requirements.txt”

## 2. Introduction to the codes files

1. The folder './TC_EIVD/':

The file contains codes for three precipitation data accuracy assessment methods: In-situ verification, MTC and EIVD
  a. 'mtc.py', 'mtc_rainfall_classification.py' are the code files used to perform  MTC analysis. Rainfall_classification represents classification based on rainfall levels.
  b. 'eivd.py','eivd_rainfall_classification.py' are the code files used to perform  EIVD analysis. Rainfall_classification represents classification based on rainfall levels.
  c.'station_statistics_point.py', 'station_statistics_grid.py', 'station_rainfall_classification.py' are the code files used to perform  in-situ verification analysis. Grid represents grid-based analysis, and point represents point-based analysis.Rainfall_classification represents classification based on rainfall levels.
  d. The code starting with "climate" represents the accuracy analysis results of each method under different climates.The code at the beginning of "landuse" represents the accuracy analysis results of each method under different land use types.

2. The folder './DNN/':

   This folder includes the training and evaluation processes of different models on different datasets.The codes in different datasets are mainly different in parameter settings, and the rest are generally the same. Here we take imerg_f7 as an example.

   - the './DNN/imerg_f7/' folder: 
   'config.py' is the configuration code, including data set specification, output directory, data processing configuration, SHAP analysis configuration, hyperparameter configuration, data partitioning configuration, training configuration, etc.
  'dataset.py' is the code for data processing and preprocessing.
  'evaluator.py' is the code for model evaluation.
  'main.py' is the code for running the entire program.
  'model.py' is the code for the DNN model.
  'shap_analyzer.py' is the code for SHAP analysis of the model.
  'trainer.py' is the code for model training.
  'visualizer.py' is the code for visualizing the results.

   - the './DNN/some preprocess/' folder: 'pearson.py' is the code for calculating the correlation coefficient between datasets. 'rep.py' is the code for calculating the spatial representation of auxiliary variables. 'world_clip.py' is the code for clipping the world using the continents shp file.

## 3. Required Datasets
1. Remotely sensed precipitation data and site-based precipitation data
2. Land use and climate data
3. World continent data
3. Auxiliary variable data - This experiment uses topography, vegetation, drought index, soil texture, etc. You may consider adding other variables to enrich the experiment.

## 4. Notes
1. The three assessment methods are independent of each other. The climate and land use type analyses are segmented based on the global analysis.
2. Note that the SHAP analysis is performed on the auxiliary variables added separately and should not be included in feature engineering.
