# Paper Notes

## Combining satellite imagery and machine learning to predict poverty
**Authors:** Neal Jean, Marshall Burke, Michael Xie, W. Matthew Davis, David B. Lobell, Stefano Ermon  
**Year:** 2016  
**Venue:** Science (Vol. 353, Issue 6301, pp. 790–794)

### Problem
Reliable, fine-grained economic livelihood data is critically scarce across the developing world. Between 2000 and 2010, 39 out of 59 African countries conducted fewer than two nationally representative surveys from which poverty measures could be constructed (14 conducted zero). Conventional household survey collection is financially and logistically prohibitive (estimated in hundreds of billions of USD over 15 years for UN SDG targets).

While nighttime lights (DMSP/VIIRS) are widely used as an economic proxy, they exhibit severe floor effects: in impoverished communities living near or below the international extreme poverty line ($1.90/day), nighttime luminosity is near zero and shows almost no variance. Daytime satellite imagery provides rich visual signals (infrastructure, roof materials, agricultural plots), but ground truth survey labels are too scarce (only hundreds of clusters per country) to train deep convolutional neural networks directly without severe overfitting.

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
2. **Automated Feature Engineering:** Generated several thousand quantitative metrics from CDR logs per user, capturing communication volume, call/SMS intensity, timing/seasonality, directional balance, social network topology (degree centrality, cluster coefficients), and geospatial mobility/migration derived from cell tower tower hops.
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


# Paper Notes

## Measuring Changes in Poverty with Deep Learning and Satellite Images
**Authors:** Lukas Kondmann, Xiao Xiang Zhu[cite: 89]  
**Year:** 2020[cite: 89]  
**Venue:** ICLR Practical ML for Developing Countries Workshop[cite: 89]  

### Problem
Transfer learning aur daytime satellite imagery se kisi location ki static wealth level predict karna pehle demonstrate ho chuka hai (jaise Jean et al., 2016)[cite: 89]. Lekin policy evaluation, poverty alleviation programs, aur interventions ka asar samajhne ke liye time ke saath hone wale badlav (longitudinal changes) ko track karna zaroori hai[cite: 89]. Yeh study yeh evaluate karti hai ki kya transfer learning pipeline satellite data se temporal changes aur economic development ko reliably measure kar sakti hai[cite: 89, 90].

### Method
- **Data:**
  - Landsat 7 ke yearly median RGB composites (1 km x 1 km resolution)[cite: 90].
  - DMSP-OLS annual nighttime lights (3 classes: low, medium, high)[cite: 90].
  - Ground truth ke roop mein Rwanda ke Demographic and Health Surveys (DHS) 2005, 2010, aur 2015 ke asset wealth scores[cite: 90].
- **Pipeline:**
  - ImageNet-pretrained ResNet-50 ko daytime Landsat 7 images se nightlight categories predict karne ke liye fine-tune kiya[cite: 90].
  - Extracted feature representations par 2005 DHS survey ke asset wealth index ko predict karne ke liye Ridge Regression fit kiya[cite: 90, 91].
  - Is trained setup ko out-of-year predictions (2010 aur 2015) ke liye evaluate kiya gaya aur cell-level/province-level growth differences calculate kiye gaye[cite: 91].

### Key Results
- **Static vs. Temporal Performance:** Model out-of-year wealth levels toh accurately predict kar leta hai ($R^2 = 0.69$ in 2010, $0.57$ in 2015)[cite: 91].
- **Change Detection Failure:** 2005 se 2015 ke dauran Rwanda mein national poverty reduction ($0.18$) aur strong GDP growth ($1.15$) hua[cite: 92]. Iske bawajood transfer learning approach ne national level par average wealth change lagbhag zero ($0.00$) predict kiya[cite: 92].
- **Contradictory Direction:** Kigali City aur Southern Province jaise growing regions mein model ne negative wealth change predict kiya, jabki ground-truth surveys aur nighttime luminosity dono mein significant positive growth dekha gaya tha[cite: 92, 93].

### What I Will Replicate
- **Longitudinal Target Benchmarks:** Multi-period survey data (DHS waves) ko use karke check karna ki kya models sirf cross-sectional levels pe overfit ho rahe hain ya actual growth signals capture kar rahe hain[cite: 9, 89, 90].
- **Nighttime Radiance Validation:** Nightlight time series (VIIRS) ko an independently verified macroeconomic trend tracker ke roop mein maintain karna, kyunki luminosity continuous temporal trends ko optic composites ke comparison mein zyada stably capture karti hai[cite: 11, 91, 92].

### What I Will Improve
- **Multi-Spectral Bands vs. Sirf RGB:** Kondmann & Zhu ne sirf 3 visible RGB bands use kiye the[cite: 90]. Hum Landsat 8/9 aur Sentinel data se multi-spectral indices (NDVI for vegetation, NDBI for built structures, MNDWI for water) extract karenge, jisse infrastructure aur agriculture ka zyada dynamic signal mile[cite: 14, 15].
- **Dedicated Temporal Architecture:** Single-frame CNN transfer learning ke badle Temporal Fusion Transformer (TFT) aur engineered lag/rolling features (1, 3, 6, 12 months) ka use karenge jo specifically time-series dynamics ke liye design kiye gaye hain[cite: 21, 36, 37].
- **Causal Validation Layer:** Natural experiments (Difference-in-Differences aur Synthetic Control) implement karenge taaki verify ho sake ki model macro shocks (jaise fuel subsidy changes) ke true directional impact ko detect kar pata hai ya nahi[cite: 25, 26, 28].

### Open Questions
- **Spatial Resolution Bottlenecks:** Kya moderate-resolution pixels (30m Landsat) subtle livelihood changes (jaise better roofing material ya paving) ko miss kar dete hain, aur kya higher-resolution Sentinel data se yeh gap bridge ho sakta hai[cite: 14, 94]?
- **Temporal Normalization:** Multi-year satellite imagery mein sensor degradation aur seasonal atmospheric noise ko normalize karne ka best approach kya hai taaki model visual artifacts ko false economic decline na samjhe[cite: 13, 90]?