# Foundational Paper Notes

---

## Combining satellite imagery and machine learning to predict poverty
**Authors:** Neal Jean, Marshall Burke, Michael Xie, W. Matthew Davis, David B. Lobell, Stefano Ermon  
**Year:** 2016  
**Venue:** Science (Vol. 353, Issue 6301, pp. 790–794)

### Problem
Reliable, fine-grained economic livelihood data is critically scarce across the developing world. Between 2000 and 2010, 39 out of 59 African countries conducted fewer than two nationally representative surveys from which poverty measures could be constructed (14 conducted zero). Conventional household survey collection is financially and logistically prohibitive (estimated in hundreds of billions of USD over 15 years for UN SDG targets).

While nighttime lights (DMSP/VIIRS) are widely used as an economic proxy, they exhibit severe floor effects: in impoverished communities living near or below the international extreme poverty line ($1.90/day), nighttime luminosity is near zero and shows almost no variance. Daytime satellite imagery provides rich visual signals (infrastructure, roof materials, agricultural plots), but ground-truth survey labels are too scarce (only hundreds of clusters per country) to train deep convolutional neural networks directly without severe overfitting.

### Method
The authors solve the extreme label scarcity problem via a three-step transfer learning pipeline:

1. **Feature Pretraining:** Initialize a CNN on the ImageNet dataset (1,000 object classes) to learn low-level visual representations (edges, textures, spatial gradients).
2. **Proxy Task Transfer Learning:** Fine-tune the CNN on a novel intermediate task: predicting nighttime light intensities (discretized into low, medium, and high bins) directly from daytime satellite imagery tiles (sourced via Google Static Maps). Because daytime imagery and nighttime illumination are globally abundant and paired, this provides an inexpensive, scalable training dataset. Through this proxy task, internal filters automatically learn semantic landscape features (urban areas, road networks, waterways, farmland) without human supervision.
3. **Survey Calibration via Ridge Regression:** Extract feature representations (4,096 dimensions) from the penultimate layer of the trained CNN for each survey cluster. Train an L2-regularized Ridge Regression model on these extracted features to predict:
   - **Consumption Expenditure:** Cluster-level mean per capita daily consumption from the World Bank Living Standards Measurement Study (LSMS).
   - **Asset Wealth Index:** Cluster-level mean asset index from USAID Demographic and Health Surveys (DHS) (first principal component of durable asset ownership).

### Key Results
- **Predictive Performance ($r^2$):**
  - **Consumption Expenditure:** Cross-validated models explain 37% to 55% of variation across 4 countries (Nigeria: $r^2 = 0.42$, Tanzania: $r^2 = 0.55$, Uganda: $r^2 = 0.41$, Malawi: $r^2 = 0.37$).
  - **Asset Wealth Index:** Explains 55% to 75% of variation across 5 countries (Nigeria: $r^2 = 0.69$, Tanzania: $r^2 = 0.58$, Uganda: $r^2 = 0.56$, Malawi: $r^2 = 0.55$, Rwanda: $r^2 = 0.75$).
  - **Pooled Models:** Explain 44% to 59% of overall variation ($r^2 = 0.45$ for consumption, $r^2 = 0.58$ for assets).
- **Outperforming Nightlights:**
  - Below 1x poverty line: Outperforms nightlights in 81.3% of trials ($+0.04$ mean $r^2$).
  - Below 2x poverty line: Outperforms nightlights in 98.5% of trials ($+0.10$ mean $r^2$, an 81.2% relative gain).
  - Below 3x poverty line: Outperforms nightlights in 99.5% of trials ($+0.12$ mean $r^2$, a 54.2% relative gain).
- **Cross-Border Generalization:** Models trained in one country transfer effectively to unobserved neighboring countries (e.g., testing on Rwanda with models trained elsewhere yields $r^2 = 0.55–0.66$ vs $0.75$ in-country).
- **Spatial Privacy Invariance:** Predictions remain robust despite surveys incorporating up to 5–10 km of random GPS noise to preserve respondent anonymity.
- **Placebo Test:** Randomly permuting daytime images across survey clusters over 1,000 trials reduces $r^2$ to near zero, verifying the signal is genuine and non-spurious.

### What I Will Replicate
- **Nightlights Baseline:** Implement a baseline mapping VIIRS monthly nightlight radiance against economic output (reproducing the primary benchmark of Jean et al.).
- **Multi-Spectral Feature Extraction:** Ingest optical bands (Landsat 8/9, Sentinel-2) alongside nightlights, computing key spectral indices:
  - NDVI (vegetation/agricultural health)
  - NDBI (built-up density/urbanization)
  - MNDWI (water body distribution)
- **Patch Extraction & CNN Embeddings:** In Phase 3, extract fixed-size spatial image patches ($64 \times 64$) and encode them using a CNN to extract spatial feature embeddings that augment tabular time-series features.

### What I Will Improve
- **Longitudinal Nowcasting vs. Static Cross-Section:** Jean et al. evaluate spatial variation at a single point in time due to uncertain imagery timestamps. My system builds a monthly/quarterly longitudinal panel (2013–2024) via Google Earth Engine to predict changes over time and nowcast economic trends up to 4 quarters ahead.
- **Causal Validation Layer:** Jean et al. rely exclusively on associative correlation ($r^2$). My project builds an explicit causal validation module evaluating known macro shocks (e.g., Nigeria's 2023 fuel subsidy removal and Naira devaluation) using Difference-in-Differences (DiD) and Synthetic Control Methods to verify that the model captures true economic impacts rather than spurious geographic correlations.
- **Strict Temporal Validation:** Implement expanding-window temporal cross-validation rather than random k-fold splits, strictly eliminating lookahead bias and temporal leakage.
- **Production Architecture & MLOps:** Package the pipeline into a FastAPI prediction service, automated weekly data ingestion using GitHub Actions, model drift monitoring (Kolmogorov-Smirnov tests), and an interactive Streamlit dashboard.

### Open Questions
- **Spatial Aggregation Bias (MAUP):** Survey ground truths are point clusters (villages/wards), whereas regional GDP statistics are aggregated to Admin-1 (state) or national levels. What is the most effective aggregation strategy that avoids ecological fallacy?
- **Cloud Cover Gaps in Equatorial Regions:** Coastal areas (such as southern Nigeria) suffer from high cloud contamination during rainy seasons. How effectively can QA pixel cloud masking and median monthly compositing maintain temporal stability without synthetic interpolation?
- **High-Frequency Shock Sensitivity:** Asset indices capture long-term structural wealth, while macro shocks (like fuel subsidy removal) immediately contract short-term disposable consumption. Will monthly nightlights and multi-spectral indices respond rapidly enough to reflect immediate macro contractions?

---

## Predicting poverty and wealth from mobile phone metadata
**Authors:** Joshua Blumenstock, Gabriel Cadamuro, Robert On  
**Year:** 2015  
**Venue:** Science (Vol. 350, Issue 6264, pp. 1073–1076)

### Problem
Reliable, quantitative economic data is critically scarce in developing economies, severely hampering policy design, resource allocation, and poverty targeting. In much of Africa, national economic production statistics can be off by as much as 50%, and spatially disaggregated small-area data rarely exists. 

Traditional nationwide data collection methods (national censuses and in-person household surveys like DHS) are prohibitively slow and expensive—typically costing over $1,000,000 and requiring 12 to 18 months to administer. Because of long lag times (such as Angola waiting 44 years between censuses), policymakers operate with outdated information. However, mobile phone adoption is ubiquitous across developing countries (3.4 billion users globally at the time), generating passive, fine-grained digital footprints.

### Method
The authors develop a machine learning methodology to infer an individual subscriber's wealth directly from their past phone transaction metadata, then scale those inferences across millions of subscribers to build nationwide, microregional wealth maps:

1. **Multi-Source Data Ingestion:**
   - **Call Detail Records (CDRs):** Billions of anonymized transactions (calls, SMS) from 1.5 million subscribers on Rwanda's primary telecom operator (May 2008 – May 2009).
   - **Phone Survey Data:** Administered a 75-question phone survey across a geographically stratified random sample of 856 subscribers to build a ground-truth Composite Wealth Index (first principal component of asset and housing indicators).
   - **Government Validation Data:** Nationally representative Demographic and Health Surveys (DHS) from 2007 (7,377 households) and 2010 (12,792 households).
2. **Automated Feature Engineering:** Generated several thousand quantitative metrics from CDR logs per user, capturing communication volume, call/SMS intensity, timing/seasonality, directional balance, social network topology (degree centrality, cluster coefficients), and geospatial mobility/migration derived from cell tower hops.
3. **Model Selection & Regularization:** Fitted Elastic Net regularized regression (combining L1 lasso sparsity and L2 ridge stability) using k-fold cross-validation to prevent overfitting on the 856-survey sample. Evaluated non-linear tree ensembles for comparison.
4. **Out-of-Sample Geographic Scaling:** Applied the trained model to the remaining 1.5 million Rwandan subscribers and mapped the geographic wealth distribution across 2,148 administrative cells and high-resolution Voronoi partitions down to groups of 5–15 households.

### Key Results
- **Individual Wealth Prediction:** Cross-validated predicted wealth correlates strongly with actual survey wealth at $r = 0.68$.
- **Binary Asset Ownership Classification (AUC):**
  - Electricity in household: $\text{AUC} = 0.85$
  - Television ownership: $\text{AUC} = 0.84$
  - Refrigerator ownership: $\text{AUC} = 0.88$
  - Motorcycle ownership: $\text{AUC} = 0.67$
- **Poverty Targeting Accuracy:** Identifying subscribers living in extreme relative poverty yielded an $\text{AUC} = 0.72$ (5th percentile threshold) and $\text{AUC} = 0.81$ (25th percentile threshold).
- **Validation Against Independent Government Surveys (DHS 2010):**
  - District-level average wealth (30 districts): $r = 0.916$ across all households, and $r = 0.917$ among phone-owning households.
  - Village / DHS cluster-level wealth (492 clusters): $r = 0.79$.
  - District electrification rates: $r = 0.93$.
  - Urban grid-cell electricity vs. nightlights in Kigali: $r = 0.58$.
- **Temporal Responsiveness:** Wealth predictions based on 2009 phone records matched the actual 2010 DHS wealth distribution more accurately than the official 2007 DHS survey did, demonstrating value for interim intercensal tracking.
- **Cost & Speed:** The phone survey cost $12,000 and took 4 weeks to complete, compared to $1M+ and 12–18 months for standard government household surveys.

### What I Will Replicate
- **Aggregated Human Activity Proxies:** Utilize passive digital traces of human presence and capital intensity. While raw telecom CDRs are proprietary and inaccessible, nighttime radiance (VIIRS) and urban built-up surface reflectance (NDBI/Landsat) serve as direct public proxies for electricity access, human activity density, and infrastructure concentration.
- **High-Dimensional Feature Engineering & Regularization:** Extract combinatorial temporal lags (1, 3, 6, 12 months), rolling statistics (mean, std), and YoY rates of change from satellite bands, utilizing regularized algorithms (L2 Ridge, Elastic Net, and gradient-boosted trees) with strict cross-validation to select non-redundant predictors.
- **Validation Against DHS Sub-National Microdata:** Benchmark regional model predictions against ground-truth DHS cluster wealth indicators for Nigeria, Rwanda, and Kenya.

### What I Will Improve
- **Open-Access & Sovereign Scalability:** Blumenstock et al. rely on proprietary CDR databases controlled by private telecom operators, presenting commercial barriers, strict NDAs, and significant privacy/surveillance concerns. My system relies entirely on open-access, publicly available satellite constellations (VIIRS, Landsat 8/9, Sentinel-2 via Google Earth Engine), allowing sovereign, fully reproducible deployment across borders without private data silos.
- **Longitudinal Nowcasting & Temporal Panel Structure:** The paper's primary focus is predicting a cross-sectional wealth distribution at a single period (2008–2009). My project structures a continuous monthly/quarterly longitudinal panel (2013–2024), predicting dynamic economic fluctuations up to 4 quarters ahead.
- **Causal Evaluation of Macro Policy Shocks:** The CDR approach is purely predictive and associational. My pipeline integrates a formal causal validation layer (Difference-in-Differences and Synthetic Control Methods) to test whether predicted indicators detect real economic interventions (such as fuel subsidy cuts and currency shocks) rather than static urbanization biases.
- **Full-Stack Automated Deployment:** Blumenstock et al.'s analysis is an offline academic batch study. My project implements a live production architecture (FastAPI endpoint, Docker container, automated GitHub Actions pipeline, drift monitoring, and Streamlit UI).

### Open Questions
- **Capturing the Excluded / Unconnected:** Just as mobile metadata under-represents individuals without cellphones or in low-coverage rural zones, optical satellite imagery struggles with informal economic transactions and cloudy coastal regions. How do we prevent systematic bias against non-electrified or agrarian communities?
- **Feature Convergence:** Blumenstock observed that mobility features predicted physical transport assets (motorcycles) while network topology predicted overall wealth. What specific spectral combinations (e.g., NDVI $\times$ NDBI interactions or seasonal radiance amplitude) serve as the best satellite analogues for disposable consumption vs. long-term physical assets?

---

## Measuring changes in poverty with deep learning and satellite images
**Authors:** Lukas Kondmann, Xiao Xiang Zhu  
**Year:** 2020  
**Venue:** ICLR Practical ML for Developing Countries Workshop

### Problem
Transfer learning using daytime satellite imagery has shown strong success in predicting static levels of local poverty across geography (e.g., Jean et al., 2016). However, the critical task for policymakers—evaluating the impact of anti-poverty interventions, policy changes, and economic shocks over time—requires measuring **temporal changes** in poverty, not merely cross-sectional differences. 

This paper investigates whether standard transfer learning pipelines can accurately measure longitudinal changes in local economic wealth over a 10-year period (2005–2015) in Rwanda, one of Africa’s fastest-growing economies during that era.

### Method
- **Data Ingestion:**
  - **Daytime Optical Imagery:** Yearly median cloud-free composites of Landsat 7 (1 km $\times$ 1 km resolution, RGB channels).
  - **Nightlight Targets:** NOAA DMSP-OLS annual nighttime luminosity composites (categorized into low, medium, and high classes).
  - **Ground Truth Survey Data:** Rwanda Demographic and Health Surveys (DHS) from 2005, 2010, and 2015 cluster-level household asset wealth scores.
- **Pipeline Implementation:**
  - Fine-tuned an ImageNet-pretrained ResNet-50 network to classify nightlight intensity classes from daytime Landsat 7 imagery across 65,000 regional training tiles.
  - Extracted deep visual features from the trained CNN and fitted an L2-regularized Ridge Regression model on 2005 DHS cluster asset wealth.
  - Deployed the model out-of-year across Rwanda at the nightlight cell level for 2010 and 2015 to compute absolute predicted growth rates relative to 2005 baseline levels.
  - Benchmarked predictions against changes in raw nighttime lights (DMSP-OLS) and official income/poverty reduction statistics from the National Institute of Statistics of Rwanda (NISR).

### Key Results
- **Failure in Temporal Change Detection:** Despite explaining cross-sectional wealth levels in unseen years ($R^2 = 0.69$ in 2010, $R^2 = 0.57$ in 2015), the transfer learning model **predicted an average national wealth change of exactly 0.00** between 2005 and 2015.
- **Severe Inconsistency with Macro Ground Truth:** During this 10-year window, Rwanda experienced an 18% reduction in national poverty and 115% real GDP growth. While raw nightlights captured positive growth across all periods ($p < 0.001$), daytime transfer learning failed to capture any positive trend.
- **Contradictory Sub-National Dynamics:** The model predicted statistically significant economic *declines* in Kigali City ($-0.04$) and the Southern Province ($-0.04$, $p < 0.001$), directly contradicting independent survey and nighttime luminosity data showing large economic expansions.
- **Root Cause Identified:** Daytime satellite imagery composites remain visually static over time. Macroeconomic improvements often manifest as subtle structural or material upgrades (e.g., higher-quality roofing, maintenance, internal household assets) that do not create drastic spatial visual differences at 30-meter optical resolutions.

### What I Will Replicate
- **Longitudinal Target Benchmarks:** Use multi-year DHS survey waves and World Bank data to evaluate whether model predictions track real temporal changes or simply overfit to static spatial geographic rankings.
- **Nighttime Radiance Validation:** Maintain nighttime lights (VIIRS) as an independent continuous signal, recognizing that luminosity captures dynamic infrastructure and electrification trends more reliably over time than optical daytime RGB tiles alone.

### What I Will Improve
- **Multi-Spectral Bands vs. Visible RGB Only:** Kondmann & Zhu restricted input imagery to 3-band visible RGB (Landsat 7). My project incorporates multi-spectral indices (NDVI for agricultural output, NDBI for built-up expansion, MNDWI for surface water) from Landsat 8/9 and Sentinel-2, capturing richer environmental and infrastructure signals.
- **Dedicated Temporal Architectures:** Instead of treating each year as an isolated static image classification problem, my system employs explicit time-series modeling (lagged feature engineering with XGBoost, progressing to Temporal Fusion Transformers) designed for temporal dynamics and multi-horizon forecasting.
- **Causal Validation Framework:** Rather than relying solely on descriptive trend comparisons, I implement formal quasi-experimental causal validation (Difference-in-Differences and Synthetic Control Methods) around clear macro shocks (e.g., Nigeria's 2023 fuel subsidy reform) to verify whether predicted contractions represent genuine causal responses.

### Open Questions
- **Sensor Resolution Bottlenecks:** Can 10-meter to 30-meter public satellites detect subtle livelihood improvements in developing regions without auxiliary mobility or commercial high-resolution data?
- **Atmospheric & Sensor Drift Normalization:** How do we separate real temporal decline in economic activity from optical sensor degradation or inter-annual cloud-cover artifacts in multi-year composites?

---


# Research Notes: Subnational Economic Nowcasting via Multi-Sensor Earth Observation

## 1. Mathematical Formulation

### 1.1 Remote Sensing Feature Space
Let $i \in \{1, \dots, N\}$ index administrative regions (States $N=37$, LGAs $N=775$) and $t$ index temporal years. The primary feature vectors extracted via Google Earth Engine zonal reductions are defined as:

1. **Nighttime Luminosity:**
   $$\text{log\_viirs}_{i,t} = \ln\left(1 + \frac{1}{|A_i|} \sum_{p \in A_i} \text{VIIRS}_{p,t}\right)$$
   where $\text{VIIRS}_{p,t}$ represents the annual median Day/Night Band (DNB) radiance ($nW \cdot cm^{-2} \cdot sr^{-1}$) over spatial footprint $A_i$.

2. **Normalized Difference Vegetation Index (MODIS 250m):**
   $$\text{NDVI}_{i,t} = \frac{1}{|A_i|} \sum_{p \in A_i} \left( \frac{\text{NIR}_{p,t} - \text{Red}_{p,t}}{\text{NIR}_{p,t} + \text{Red}_{p,t}} \right)$$

3. **Sentinel-2 L2A BOA Multi-Spectral Indices (10m–20m):**
   * **Normalized Difference Built-Up Index (NDBI):**
     $$\text{NDBI} = \frac{\text{SWIR1} - \text{NIR}}{\text{SWIR1} + \text{NIR}} = \frac{B_{11} - B_8}{B_{11} + B_8}$$
   * **Modified Normalized Difference Water Index (MNDWI):**
     $$\text{MNDWI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1}} = \frac{B_3 - B_{11}}{B_3 + B_{11}}$$
   * **Bare Soil Index (BSI):**
     $$\text{BSI} = \frac{(B_{11} + B_4) - (B_8 + B_2)}{(B_{11} + B_4) + (B_8 + B_2)}$$

---

## 2. Spatial Econometric Specification

### 2.1 Queen Contiguity Weight Matrix ($W$)
To account for geographic proximity across contiguous subnational boundaries:
$$w_{ij} = \begin{cases} 1 & \text{if } \text{bnd}(i) \cap \text{bnd}(j) \neq \emptyset, \, i \neq j \\ 0 & \text{otherwise} \end{cases}$$
Row-standardized transformation:
$$W_{ij}^* = \frac{w_{ij}}{\sum_{k=1}^N w_{ik}}$$

### 2.2 Global Moran's $I$ Test
Spatial autocorrelation of the poverty headcount vector $y$ is computed as:
$$I = \frac{N}{\sum_{i=1}^N \sum_{j=1}^N W_{ij}^*} \frac{\sum_{i=1}^N \sum_{j=1}^N W_{ij}^*(y_i - \bar{y})(y_j - \bar{y})}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
* **Empirical Result:** $I = 0.7337$, $\mathbb{E}[I] = -0.0278$, $z = 6.42$, $p < 0.00001$.
* **Conclusion:** Rejection of the null hypothesis ($H_0$: spatial randomness). Poverty displays intense positive spatial clustering across Nigeria.

### 2.3 Spatial Autoregressive Model (SAR / 2SLS)
To capture spatial lag spillovers while overcoming endogeneity ($\text{Cov}(Wy, \epsilon) \neq 0$), we estimate the model using Spatial Two-Stage Least Squares (2SLS) instrumented by $[WX, W^2X]$:
$$y = \rho W y + X \beta + \epsilon$$

**Empirical Estimates:**
* Constant: $\alpha = 133.33$ ($p = 0.00003$)
* $\log(\text{VIIRS})$: $\beta_1 = -38.87$ ($z = -5.24, p < 0.00001$)
* MODIS NDVI: $\beta_2 = -159.76$ ($z = -3.80, p = 0.00015$)
* Spatial Lag ($\rho$): $\rho = 0.1823$ ($z = 0.89, p = 0.373$)
* **Model Pseudo $R^2$:** $0.8633$
* **Anselin-Kelejian Test:** Value $= 0.016$, $p = 0.9000$ (No remaining spatial error dependence).

---

## 3. Panel Dynamics & Structural Shocks (2018–2023)

### 3.1 Entity Fixed Effects Estimation
For balanced panel data ($N=37, T=6, NT=222$):
$$y_{it} = \mu_i + \beta X_{it} + \epsilon_{it}$$
where $\mu_i$ absorbs all time-invariant, unobserved state structural fixed effects (topography, coastlines, capital endowment).

* **Poolability Test:** $F(36, 184) = 89.04$ ($p < 0.00001$), formally rejecting naive pooled OLS.

### 3.2 The Niger Delta Hydrocarbon / Flaring Discontinuity
Panel trajectory analysis between 2018 and 2023 reveals an atypical decline in nighttime radiance across the South-South geopolitical zone:
* Bayelsa: $-55.4\%$
* Akwa Ibom: $-42.8\%$
* Delta: $-33.9\%$
* Rivers: $-31.4\%$

**Qualitative Context:** Unlike agrarian or commercial states where night lights directly correlate with household grid electrification, Niger Delta radiance is dominated by upstream industrial gas flaring terminals. Flare capture initiatives and pipeline interruptions produce sharp radiance contractions that reflect hydrocarbon operational shifts rather than sudden drops in household welfare.

---

## 4. Cross-Validation Protocols

Standard $K$-Fold cross-validation suffers from spatial data leakage due to spatial autocorrelation ($I = 0.7337$). To guarantee generalization across unobserved geographical domains, we adopt **Spatial Leave-One-Geopolitical-Zone-Out Cross-Validation**:

$$\text{Test Sets } Z \in \{\text{North Central}, \text{North East}, \text{North West}, \text{South East}, \text{South South}, \text{South West}\}$$

* **Spatial CV $R^2$:** $0.805$ – $0.810$
* **Spatial CV RMSE:** $\sim 10.6\%$ points
* **Spatial CV MAE:** $\sim 8.4\%$ points