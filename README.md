# easy-sdm

<!-- ![easy_sdm](./docs/imgs/logo_easy_sdm.png) -->

<p align="center">
<img style="vertical-align:middle" src="https://github.com/AI-Unicamp/easy-sdm/blob/d23c6b60acbb08fc574d2b3b1919b36b706e688c/docs/imgs/logo_easy_sdm.png" />
</p>
<h1 align="center">
</h1>

easy-sdm is the MSc final result from [Matheus Gustavo Alves Sasso](https://github.com/math-sasso) at UNICAMP (State University of Campinas).



## 1) Introduction

This project is a SDM (Species Distribution Modelling) focused on Brazilian vegetal species applied to agriculture. To create the species distributions species occurrences data were extracted from GBIF and environment data were extracted from the bases [ENVIREM](https://envirem.github.io/), [BIOCLIM](https://www.worldclim.org/data/worldclim21.html) and [Soilgrids](https://soilgrids.org/).

The data required for this task is a combination of geospatial occurrences of species found in Global Biodiversity Information Facility (GBIF) with environmental data available as 2D raster images, in which we prepossessed and stacked t creating a 3D space representing the Brazilian environment that we call **br-env**.

We modeled SDMs in a pipeline that combines One Class Support Vector Machine (OCSVM) to generate pseud-absences that crates a binary classification problem scenario and ensemble classifiers that identify the suitability of a species in the pixels of the Brazilian map in the (0,1) interval. Two objective metrics are proposed to assess the ability of models to identify a region’s suitability for species, Area Under the Receive Operation Curve (AUC) and True Skills Statistic (TSS), evaluated in cross-validation to guarantee that results stable independently of the geographical split. We also applied the Variance Inflation Factor (VIF) algorithm to evaluate if a reduced number of environment variables were sufficient to enable the classification models to separate the two classes.

## 2) br-env

**br-env** is a standardized database framed on the Brazilian territory that combines data from [ENVIREM](https://envirem.github.io/), [BIOCLIM](https://www.worldclim.org/data/worldclim21.html) and [Soilgrids](https://soilgrids.org/). It consists in a 3D numpy array  with dimensions [K, W, H] and a list artifact containing the variables’ names in the order they were placed in the matrix. K is the number of environment variables, M is the number of matrix widths, and H is the matrix height. We consider **br-env** as a major contribution of our work once it can be used as a backbone for SDMs modeled with diffent algorithms and species, being relevant to speed up ecological studies in Brazil.
To download **br-env** please go[here](https://drive.google.com/drive/folders/1M2txhMKZif6dnJAt-xFREuMwOMux4BLz?usp=sharing)
