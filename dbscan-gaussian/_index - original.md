---

layout: default

title: Road Collisions (DBSCAN and Gaussian Mixture Models)

permalink: /dbscan-gaussian/

---

# This project is in development

## Goals and objectives:

The business objective is to show how density-based and model-based clustering can help a county highway authority decide where to concentrate a limited road-safety budget, and, just as importantly, to show when a simpler method such as K-Means is the wrong tool for the job.

A highway authority has a fixed annual budget for engineering interventions: junction redesign, speed management, signing and lining, anti-skid surfacing. It cannot investigate every location where a collision has been recorded, so it needs a short list of places to examine first. The data available is a year of police-reported injury collisions, each with a map location and a severity (fatal, serious or slight). This project uses the 3,896 collisions recorded in Kent in 2023 to find hotspots, and then asks whether those hotspots still concentrate collisions in 2024 (3,344 collisions), which is when the same authority would actually be spending the budget.

The [K-Means Clustering](/k-means-clustering/) project was previously the only clustering method in this portfolio, and collision locations are exactly the kind of data that stretches its assumptions. K-Means expects compact, roughly spherical clusters, forces every observation into a cluster, and needs K to be chosen in advance. Collisions cluster around junctions and along roads rather than in round blobs, many collisions are isolated events on quiet roads that belong to no hotspot at all, and there is no natural number of hotspots to specify. Two alternatives relax these assumptions in different ways:

* **Gaussian Mixture Models (GMM)** allow clusters of different sizes, elliptical shapes and orientations, and give each observation a probability of belonging to each cluster rather than a single hard assignment. They still assign every observation to a cluster and still need the number of components to be chosen.
* **DBSCAN** defines clusters as regions of high point density. It needs no K, finds clusters of arbitrary shape, and labels points in sparse regions as noise rather than forcing them into a cluster, but it relies on a single density threshold for the whole dataset.

The analysis is structured around three questions:

* **Where do each method's assumptions hold, and where do they fail?** A synthetic benchmark of four small datasets with known structure (interleaved crescents, stretched ellipses, clusters of very different density, and clusters buried in background noise) isolates what each method assumes, using the true labels to score each one. GMM and DBSCAN are then run on the Wheat Seeds data from the K-Means project, allowing a like-for-like comparison with its Adjusted Rand Index of 0.773 and its 91.90% agreement with the true varieties.
* **How are each method's parameters chosen without labels, and how far can those rules be trusted?** Real hotspot data has no true labels to tune against. The silhouette score (K-Means), BIC and AIC (GMM) and the k-distance curve with a sensitivity grid (DBSCAN) are each audited on the synthetic data, where the right answer is known, before being applied to the collision data.
* **Do the hotspots found in one year still capture collisions in the next, for the same amount of ground to investigate?** This is the question the highway authority actually cares about. It is answered with an out-of-time test: each method flags locations using 2023 data alone, every method is given the same area to flag, and the share of 2024 collisions falling inside the flagged ground is measured and compared with a practitioner benchmark (a simple grid of 500 m squares ranked by collision count). Bootstrap resampling quantifies how much of any difference could be chance, and the whole procedure is repeated on an earlier pair of years to check that the result is not specific to one year.

Two design choices follow from the [Data Science Workflow](/data-science-workflow/) page. First, the out-of-time success measure was agreed at the scoping stage, before any model was fitted to the collision data, to avoid selecting a metric after the fact that favours a preferred method. Second, because collision hotspots have no ground-truth labels, and because intrinsic scores such as the silhouette reward compact shapes rather than useful ones, the models are judged on held-out data from a later period. This mirrors the time-ordered splits that the Workflow page uses for time-dependent problems.

A decision-maker would use the results in three ways: to draw up a ranked short list of candidate sites for detailed collision-record review and site visits; to judge how much of the county's collision problem a hotspot approach can realistically cover for a given amount of ground; and to choose between methods on evidence about next year rather than on how tidy this year's clusters look.

The framing also raises questions that go beyond the model, and the [Ethics in Applied Data Science](/data-science-ethics/) page provides the context for them. A hotspot list directs scarce attention, so there is an accountability question about what happens to the places it leaves out: a collision labelled as noise is not a safe location, only an isolated one. Ranking sites by collision count and ranking them by the number of fatal and serious collisions can give different priorities, and choosing between them is a value judgement that should be made explicitly. A single density threshold may also favour dense urban areas over rural ones, which makes the geography of the results a fairness consideration in its own right. The project therefore examines where the hotspots fall, evaluates severity separately from volume, and reports its limitations, including a change in the severity mix of the data between years, alongside its positive findings.

As with the rest of this portfolio, the aim is not to produce an operational hotspot list or to achieve business-grade predictive performance. The effects measured here are modest, and the value of the project lies in the methodology: how to choose and defend a clustering method for spatial data, and how honestly to measure whether it worked.

## Application:  

Density-based and model-based clustering extend the idea behind K-Means, which is to find structure in unlabelled data, to settings where compact, equally sized clusters are an unrealistic assumption.

**Gaussian Mixture Models** treat the data as drawn from a mixture of several Gaussian (bell-shaped) distributions, each with its own centre, spread, shape and weight. The parameters are estimated iteratively using the Expectation-Maximisation algorithm, and each observation is then given a **probability of membership** in every component. This soft assignment is the main practical advantage over K-Means: observations sitting between two groups are reported as uncertain rather than being silently forced into one. The covariance type controls how flexible each component's shape is, from spherical (equivalent in spirit to K-Means) through to full, where every component has its own elliptical shape and orientation. The number of components is typically chosen using the **Bayesian Information Criterion (BIC)** or the Akaike Information Criterion (AIC), which reward fit but penalise complexity.

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) works from a different principle. A point is a *core point* if at least `min_samples` points, including itself, lie within a distance `eps` of it. Clusters are built by linking core points that are within `eps` of one another, together with the border points that surround them, and any point that is reachable from no core point is labelled **noise**. There is no K to specify, clusters can take any shape, and outliers are handled naturally. The trade-offs are that a single `eps` must suit the whole dataset, so clusters of very different density are hard to find together, that results can be sensitive to the two parameters, and that the distance measure has to be meaningful for the data. In the [Anomaly Detection](/anomaly-detection/) project, Isolation Forest produces an anomaly score that is tuned against an explicit cost of missed and false alerts. DBSCAN's noise label is different in kind: it is a by-product of density rather than a score designed for the purpose, so it should not be read as an anomaly detector.

The three methods can be seen as a ladder of assumptions:

* **K-Means**: compact, similarly sized, spherical clusters; every point assigned; K chosen in advance.
* **GMM**: elliptical clusters of different sizes and orientations; soft memberships; every point still assigned; the number of components chosen in advance, typically by BIC.
* **DBSCAN**: clusters of any shape; no K; isolated points labelled as noise; one density threshold for the whole dataset.

No method is best in general. The right choice depends on which assumptions the data actually satisfies, and this is why the project tests each method on data where the truth is known before relying on it where it is not.

These techniques are used across many sectors, and the practical examples below show where they provide clear business value:

🚦 **Transport and Public Sector**:

**Road-safety hotspot identification**: Highway authorities cluster the locations of injury collisions to produce a short list of sites for engineering review, treating isolated collisions as noise rather than as hotspots.

**Infrastructure maintenance planning**: Councils group geocoded reports of potholes and road defects into dense clusters, so that repair crews can be scheduled by area rather than visiting reports one at a time.

**Public transport demand analysis**: Operators identify dense clusters of boarding points or journey origins from smartphone and ticketing data to inform stop placement and service frequency.

🏥 **Healthcare and Emergency Services**:

**Disease surveillance**: Public health teams cluster geocoded case reports to detect localised outbreaks, with sporadic isolated cases left as noise rather than inflating the apparent cluster.

**Emergency response planning**: Ambulance and fire services cluster historical call-out locations to decide where to position vehicles and crews, supporting shorter response times.

**Patient cohort discovery**: Analysts fit Gaussian Mixture Models to clinical measurements to identify overlapping patient subgroups, with soft membership probabilities reflecting that real patients do not always belong cleanly to one group.

🛍️ **Retail and Logistics**:

**Delivery zone design**: Logistics planners cluster delivery addresses into dense drop zones for route planning, handling remote one-off addresses separately instead of distorting the main zones.

**Footfall and site analysis**: Retailers analyse where customers dwell from anonymised location data to identify high-traffic zones for store placement and in-store layout.

**Customer segmentation**: Marketing teams use Gaussian Mixture Models to describe customer groups that overlap, using membership probabilities to weight offers rather than assigning each customer to a single segment.

🏦 **Finance and Insurance**:

**Claims geography after weather events**: Insurers cluster the locations of property claims following floods or storms to identify the affected areas, focus loss adjusters where claims are densest, and distinguish isolated claims from the event.

**Fraud ring detection**: Investigators look for dense groups of accounts or claims sharing unusual attributes, using density to surface coordinated activity while leaving unconnected cases out.

**Loss and risk modelling**: Actuaries fit mixtures of distributions to claim sizes, so that routine small claims and rarer large losses can be modelled as separate components rather than forced into one distribution.

## Methodology:  

### 1. Datasets

Three datasets are used, each for a different purpose.

* **Synthetic benchmark.** Four two-dimensional datasets of 1,500 points, generated with fixed seeds: interleaved crescents (moons), three stretched elliptical blobs (anisotropic), three blobs of very different spread (varied density), and three compact blobs with 150 uniformly scattered noise points (10%). The true labels, including which points are noise, are known, so each method can be scored directly.
* **Wheat Seeds.** The UCI Seeds dataset ([archive.ics.uci.edu/ml/datasets/seeds](https://archive.ics.uci.edu/ml/datasets/seeds)): 210 kernels, seven geometric measurements and three varieties of 70, exactly as used in the [K-Means Clustering](/k-means-clustering/) project. Features are standardised, and the labels are used only for evaluation.
* **Kent road collisions.** The Department for Transport road safety data (STATS19): police-reported injury collisions, published under the Open Government Licence v3.0 ([data.gov.uk](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)). Each record has an Ordnance Survey grid location in metres and a severity (fatal, serious or slight); **KSI** denotes fatal or serious. Kent (police force code 46) has 3,983 collisions in 2022, 3,896 in 2023 and 3,344 in 2024. No coordinates are missing in any year, and only 0.5% of 2023 collisions share an exact location with another.

### 2. Study area and data caveats

Police forces were screened for at least 3,000 collisions in each year and no more than 5% missing coordinates, which left seven. Kent was chosen because it combines a balanced urban and rural mix (51.6% urban) with a county-scale extent (4,209 km²). The Metropolitan Police area is 97% urban, leaving almost no rural background to treat as noise; West Midlands (94%) and West Yorkshire (81%) are also urban-dominated; and Police Scotland covers 88,084 km² with density that varies so widely that a single DBSCAN threshold would be expected to fit poorly.

Two caveats shape how results should be read. Only injury collisions reported to the police are recorded, so the data describes recorded collisions rather than all road risk. And the KSI share in Kent rose from 18.8% (2022) to 21.3% (2023) and 28.0% (2024) as serious collisions rose and slight collisions fell. The data cannot show why, and a change in recording practice is one possibility, so every severity result is measured against the test year's own KSI share.

### 3. Analytical approach

1. **Synthetic benchmark.** K-Means and GMM are given the true number of clusters, which is deliberately generous, so that any failure reflects a method's assumptions rather than a poor choice of K. DBSCAN uses `min_samples` = 10 and sets `eps` at the knee of the k-distance curve (each point's distance to its 10th nearest neighbour, sorted). Methods are scored with the Adjusted Rand Index (ARI), V-measure and silhouette, with DBSCAN's noise label treated as an extra group.
2. **Parameter selection without labels.** K-Means K is chosen from the elbow and silhouette (K = 2 to 8); GMM components (1 to 8) and covariance type (full, tied, diagonal, spherical) from BIC and AIC; DBSCAN `eps` from the knee, then a sensitivity grid of six `eps` multiples (0.5× to 2.0×) by four `min_samples` values. The labels are used here only to audit how reliable these rules are.
3. **Wheat Seeds bridge.** Preprocessing, seed and settings match the K-Means project, whose results are reproduced first. GMM is fitted at K = 3 for each covariance type and again with BIC-selected settings. DBSCAN uses `min_samples` = 14 (twice the number of features) with the knee `eps`, plus a grid search whose best case, which uses the labels, is reported as an upper bound. Agreement with the true varieties uses Hungarian matching, as on the K-Means page.
4. **Kent fit year (2023).** DBSCAN is fitted across nine `eps` values (100 to 1,000 m) and five `min_samples` values (5 to 20). Three DBSCAN tiers are then defined: **Site**, the setting that places the most collisions inside hotspots among those where no hotspot has a radius (centroid to furthest member) above 750 m; **Strict site** (200 m, 10); and **Area** (500 m, 10). K-Means and full-covariance GMM are fitted with as many clusters as the Site tier found, and again with K = 200.
5. **Out-of-time validation (2024).** Each method flags locations from 2023 data alone. A flagged zone is every location within 150 m of a collision in the flagged clusters, measured on a 25 m raster, with clusters flagged strongest first by collisions (or KSI collisions) per km² of footprint. Every method is given the same area: first the full area of the Site tier's zone, then 5% and 10% of the *collision-affected area* (everywhere within 150 m of any 2023 collision). The final cluster is flagged only as far as the budget allows, densest cells first. **Capture** is the share of all 2024 collisions inside the zone, **lift** compares it with random placement of the same area, and KSI capture and enrichment are reported separately. A grid of 500 m squares ranked by 2023 collision count is the practitioner benchmark. A paired bootstrap (1,000 resamples of the 2024 collisions) gives 95% intervals for each figure and for the difference between methods, and **retention** (the share of collisions inside the zone in 2024 divided by the share in 2023) measures regression to the mean.
6. **Replication.** The whole procedure, including re-selecting the Site tier, is repeated with 2022 as the fit year and 2023 as the test year.

### 4. Key methodological decisions

* **Coordinates in metres, unscaled.** Standardising would stretch the two axes unequally and make `eps` meaningless as a distance. In metres, `eps` has a physical meaning that can be reasoned about.
* **A size guard on hotspot radius.** An earlier guard limited the largest hotspot's share of all collisions, but a 243-collision conurbation with a 2.7 km radius passed it. A physical radius is the better test, because the authority investigates sites and road sections that an engineer can survey, not whole towns. The k-distance knee is reported for reference only, since a single global knee reflects the sparse rural roads rather than site-scale density.
* **Three DBSCAN scales.** Results depend on scale, so Site, Strict site and Area are all carried through validation rather than hiding the choice behind one setting.
* **Matched K, plus K = 200.** DBSCAN needs no K, so giving K-Means and GMM its cluster count is generous to them, and K = 200 tests whether a finer partition helps.
* **Equal area, not equal clusters.** The methods produce very different numbers and sizes of cluster, while area is what the authority actually spends investigating. The same footprint rule is applied to every method.
* **Out-of-time capture rather than silhouette.** The synthetic benchmark shows the silhouette rewarding compact shapes over correct structure, so hotspots are judged on whether they predict next year's collisions.
* **Ranking by density, and by severity separately.** Density measures how strongly collisions concentrate per unit of ground. Ranking by collision count and by KSI count can give different priorities, so both are reported.
* **A simple benchmark and explicit uncertainty.** Clustering must beat the simplest practitioner method to justify its added complexity, and the bootstrap prevents small differences being over-read.

All random processes use fixed seeds, so every figure reproduces exactly, and the full analysis runs on a CPU-only laptop in a few minutes.

## Results:

**Synthetic Benchmark: Where Each Method's Assumptions Hold and Fail**

K-Means and GMM were given the true number of clusters on each dataset, so the differences below reflect assumptions about shape, density and noise rather than a poor choice of K.

| Dataset | K-Means ARI (silhouette) | GMM ARI (silhouette) | DBSCAN ARI (silhouette) | DBSCAN noise |
|---|---|---|---|---|
| Moons | 0.48 (0.49) | 0.51 (0.49) | **0.93** (0.39) | 3.5% |
| Anisotropic blobs | 0.56 (0.51) | **1.00** (0.45) | 0.96 (0.47) | 2.9% |
| Varied-density blobs | 0.81 (0.63) | **0.97** (0.60) | 0.55 (0.64) | 4.7% |
| Blobs with noise | 0.84 (0.67) | 0.84 (0.60) | **0.96** (0.77) | 10.3% |

![02_synthetic_method_comparison](02_synthetic_method_comparison.png)

_Figure 1: Each method on each dataset_ 

Each method wins where its assumptions hold and fails where they do not.

* **Moons.** K-Means and GMM cut each crescent into compact halves (ARI 0.48 and 0.51), while DBSCAN follows the crescents (ARI 0.93), labelling 3.5% of points on their edges as noise.
* **Anisotropic blobs.** K-Means, which implicitly assumes equal spherical variance, mixes the stretched clusters (ARI 0.56). GMM's full covariance recovers them exactly (ARI 1.00), and DBSCAN comes close (0.96).
* **Varied-density blobs.** GMM is best (0.97) and K-Means reasonable (0.81). DBSCAN finds only two clusters (ARI 0.55), merging the diffuse cluster with a neighbouring tight one, because no single `eps` suits both densities.
* **Blobs with noise.** K-Means and GMM must place every point in a cluster, and both reach 0.84. DBSCAN reaches 0.96 and labels 10.3% of points as noise against a true 10%.

![03_synthetic_ari_vs_silhouette](03_synthetic_ari_vs_silhouette.png)

_Figure 2: Adjusted Rand Index (left) beside silhouette score (right)_ 

The silhouette, which needs no labels, often ranks methods in the opposite order to the truth. On the moons it prefers K-Means and GMM (0.49) to DBSCAN (0.39) although DBSCAN recovers the structure and they do not. On the anisotropic blobs it gives K-Means the highest score (0.51) and GMM, which is perfect, the lowest (0.45). On the varied-density blobs DBSCAN scores highest (0.64) while merging two clusters. The silhouette for DBSCAN also excludes noise points, which flatters it on the noisy dataset (0.77). Intrinsic scores reward compact shapes, which is why the collision analysis later is judged on out-of-time performance instead.

**Choosing Parameters Without Labels**

Real hotspot data has no true labels, so each method's selection rule was tested where the truth is known.

| Dataset | True K | K-Means K (silhouette) | GMM by BIC (components, covariance) | DBSCAN settings with ARI ≥ 0.8 |
|---|---|---|---|---|
| Moons | 2 | 2 | 8, full | 14 of 24 (58.3%) |
| Anisotropic blobs | 3 | 2 | 3, tied | 10 of 24 (41.7%) |
| Varied-density blobs | 3 | 3 | 3, diagonal | 8 of 24 (33.3%) |
| Blobs with noise | 3 | 2 | 5, diagonal | 21 of 24 (87.5%) |

* **K-Means (silhouette).** The maximum silhouette recovers the true K on two of four datasets and picks K = 2 on the anisotropic and noisy blobs, where the truth is 3.
* **GMM (BIC).** BIC recovers three components on the anisotropic and varied-density blobs and correctly chooses tied covariance for the anisotropic blobs, which share one covariance because a single linear transform created them. On the moons it chooses eight components to tile the crescents with Gaussians, and on the noisy blobs five (AIC chooses seven), because extra components absorb the background noise. BIC counts Gaussians, not natural clusters.
* **DBSCAN (k-distance knee).** The knee `eps` (0.13, 0.16, 0.21 and 0.12 in standardised units) recovers the structure on three of the four datasets (ARI 0.93 to 0.96) and fails on the varied-density blobs (0.55).

![07_dbscan_sensitivity](07_dbscan_sensitivity.png)

_Figure 3: DBSCAN's ARI (top) and number of clusters found (bottom) across `eps` multiples and `min_samples`_ 

The result depends heavily on the settings. On the moons it is largely stable at and above the knee `eps` (ARI 0.72 to 1.00 for `min_samples` of 10 or more) but fragments into around 30 clusters at half the knee value with `min_samples` of 5 or 10. On the anisotropic blobs it collapses to one or two clusters at 1.5 times the knee value or more. On the noisy blobs it holds across almost the whole grid. On the varied-density blobs only some of the larger `min_samples` settings (15 to 20) recover three clusters (ARI 0.83 to 0.88), and a label-free rule would not reveal which. No selection rule is reliable enough to be trusted blindly, which is why the Kent analysis reports results across a range of `eps` values.

**Wheat Seeds: Comparison with the K-Means Project**

The same preprocessing, seed and settings as the [K-Means Clustering](/k-means-clustering/) project reproduce its results exactly: ARI 0.773, silhouette 0.40 and 91.90% matched agreement with the true varieties.

| Method | Clusters | ARI | Matched agreement | Silhouette |
|---|---|---|---|---|
| K-Means (K = 3) | 3 | 0.773 | 91.90% | 0.40 |
| GMM, full covariance (K = 3) | 3 | 0.723 | 89.52% | 0.36 |
| GMM, **tied** covariance (K = 3) | 3 | **0.890** | **96.19%** | 0.38 |
| GMM, diagonal covariance (K = 3) | 3 | 0.679 | 87.62% | 0.37 |
| GMM, spherical covariance (K = 3) | 3 | 0.758 | 91.43% | 0.40 |
| GMM selected by BIC (full, K = 4) | 4 | 0.610 | 74.76% | 0.28 |
| DBSCAN at the knee `eps` (1.51, `min_samples` 14) | 1 | 0.000 | 33.33% | not defined |
| DBSCAN, best of the grid (uses the labels) | 2 | 0.388 | 56.19% | 0.47 |

![08_wheat_pca_comparison](08_wheat_pca_comparison.png)

_Figure 4: The seeds in the first two principal components (89.0% of variance) coloured by true variety and by each method_ 

The varieties overlap, with no visible gaps between them.

![09_wheat_contingency_tables](09_wheat_contingency_tables.png)

_Figure 5: Contingency tables of true variety against matched cluster for K-Means, two GMMs and DBSCAN_ 

K-Means places 193 of 210 seeds correctly. GMM with tied covariance places 202, recovering one variety perfectly (70 of 70) and misassigning eight seeds. GMM with full covariance places 188. DBSCAN puts 208 seeds in a single cluster and labels two as noise.

* **GMM helps only with the right covariance structure.** Tied covariance, one shared shape for all three varieties, raises agreement from 91.90% to 96.19% over K-Means. Full covariance does worse (89.52%), because it estimates 28 covariance parameters per component from only 70 seeds each. This fits the varieties differing mainly in position and size rather than in shape.
* **Label-free selection chose the wrong model.** BIC selects full covariance with four components (74.76% agreement), rewarding likelihood fit rather than recovery of the varieties, as it did on the synthetic moons and noisy blobs.
* **DBSCAN fails here, as its assumptions predict.** At the knee `eps` it finds a single cluster, and even the best grid setting, chosen using the labels, reaches only 56.19%. The varieties overlap with no low-density gaps to separate them, which makes this a useful example of when not to use DBSCAN.
* **Intrinsic scores barely separate the methods.** Silhouette values for the three-cluster models range only from 0.36 to 0.40, while agreement with the true varieties ranges from 87.62% to 96.19%.

**Kent Collisions: The Fit Year (2023)**

Kent recorded 3,896 injury collisions in 2023: 44 fatal, 785 serious and 3,067 slight, so 21.3% were fatal or serious (KSI). Every collision has a valid location, and only 0.5% share an exact location with another.

![12_kent_collisions_map](12_kent_collisions_map.png)

_Figure 6: All 2023 collisions by severity_ 

The map shows what makes the data awkward for K-Means: dense concentrations in the towns, long thin lines along the main roads, and a scatter of isolated collisions across the rural roads, with no natural number of clusters.

**Choosing DBSCAN's Scale**

DBSCAN was fitted across nine `eps` values (100 to 1,000 m) and five `min_samples` values (5 to 20), recording for each the number of hotspots, the share of collisions labelled noise and the radius of the largest hotspot.

![13_kent_dbscan_diagnostics](13_kent_dbscan_diagnostics.png)

_Figure 7: The k-distance curve (left) and the number of hotspots, noise share and largest hotspot radius across the grid_ 

The knee of the k-distance curve sits at 1,745 m, more than ten times the `eps` eventually chosen for the Site tier. This is the varied-density weakness from the synthetic benchmark appearing on real data: a single global distance tuned to the sparse rural roads would merge entire towns. The grid shows the consequence directly. At `min_samples` of 10, raising `eps` from 400 m to 500 m takes the largest hotspot's radius from 1,803 m to 2,731 m, and by 750 m it reaches 6,548 m.

The **Site** tier is the grid setting that places the most collisions inside hotspots among those where no hotspot has a radius above 750 m. The best-covering setting that passes is 150 m with `min_samples` 5, followed by 100 m with 5 (6.0% of collisions in hotspots) and 200 m with 10 (5.2%). At that `min_samples`, moving to 200 m raises the number of hotspots from 72 to 91 but the largest radius from 406 m to 974 m, which breaches the guard. Two further tiers are carried forward so that results can be compared across scales.

| Tier | `eps` | `min_samples` | Hotspots | Collisions in hotspots | Median hotspot size | Median radius | Largest radius |
|---|---|---|---|---|---|---|---|
| **Site** | 150 m | 5 | 72 | 13.8% | 6 | 136 m | 406 m |
| Strict site | 200 m | 10 | 15 | 5.2% | 12 | 220 m | 396 m |
| Area | 500 m | 10 | 48 | 34.9% | 17.5 | 688 m | 2,731 m |

![18_kent_dbscan_tiers](18_kent_dbscan_tiers.png)

_Figure 8: The three DBSCAN tiers mapped_ 

The Site hotspots are small and numerous. The Strict tier keeps only the densest handful, mostly around the Medway towns and north-west Kent. The Area tier covers more collisions, but its hotspots are town-sized clusters with a median radius of 688 m and a largest of 2,731 m, which is an area to be investigated rather than a site.

**The Three Methods at Matched Granularity**

K-Means and GMM were given as many clusters as the Site tier found (K = 72), and also K = 200.

![14_kent_method_maps](14_kent_method_maps.png)

_Figure 9: K-Means (K = 72), GMM with full covariance (K = 72) and the DBSCAN Site tier_ 

K-Means and GMM partition the whole county, so every collision, including the isolated ones on quiet roads, is assigned to a region, with a median of 38.5 and 42 collisions per region. DBSCAN flags 13.8% of collisions in 72 hotspots with a median of just 6, and labels the other 86.2% as noise. GMM's full covariance lets some of its regions stretch along roads, where K-Means' cells are rounder, but it still tiles the county rather than isolating hotspots.

Other fit-year findings:

* **GMM model selection.** BIC is almost flat between 30 and 60 components (lowest at 60) and rises beyond, so K = 72 sits just above the plateau and K = 200 well above it. As on the synthetic data, BIC identifies a good density model rather than a number of hotspots.
* **GMM's soft membership.** At K = 72, 20.7% of collisions have a highest membership probability below 80%. These boundary cases lie where neighbouring components overlap, and the probabilities describe which region a collision belongs to, not whether it is a hotspot.
* **Silhouette.** K-Means scores 0.44 at both K = 72 and K = 200, and GMM 0.31 and 0.35. DBSCAN's 0.78 for the Site tier is computed on the 13.8% of collisions inside hotspots only, so it cannot be compared, as the synthetic benchmark showed.

**What the Site Hotspots Look Like**

| Hotspot | Grid ref. | Collisions | KSI | Urban | Radius |
|---|---|---|---|---|---|
| #26 | TQ 7567 | 32 | 18.8% | 100% | 406 m |
| #15 | TQ 7369 | 24 | 20.8% | 100% | 358 m |
| #9 | TQ 6473 | 21 | 9.5% | 100% | 368 m |
| #41 | TQ 7667 | 15 | 13.3% | 100% | 305 m |
| #3 | TQ 7571 | 12 | 16.7% | 8.3% | 171 m |
| #7 | TQ 7467 | 12 | 25.0% | 100% | 220 m |
| #24 | TQ 5839 | 12 | 16.7% | 91.7% | 247 m |
| #1 | TQ 5473 | 11 | 27.3% | 100% | 118 m |
| #36 | TQ 5575 | 11 | 0.0% | 18.2% | 269 m |
| #51 | TQ 7462 | 10 | 40.0% | 100% | 188 m |

_Table: the ten largest Site hotspots (overall 2023 KSI share 21.3%)._

* **They are small.** The largest hotspot holds 32 collisions (0.8% of the year's total), and the median holds 6. This is a short list of specific places, not a map of the county's collision problem.
* **Volume and severity differ.** Seven of the ten largest hotspots have a KSI share below the county's 21.3%, and the highest (#51, 40%) rests on four collisions out of ten. Ranking sites by collision count and by severity gives different priorities, and with counts this small the percentages are noisy.
* **They sit mostly in urban areas, but not only there.** Eight of the ten are over 90% urban, while two (#3 and #36) are largely non-urban. From the grid references, the largest concentrations lie around the Medway towns and in Dartford and Gravesend in the north-west, with further single hotspots near Maidstone, Sittingbourne, Canterbury, Thanet, Folkestone and Tunbridge Wells. Much of the rural south and west contains few or none. DBSCAN, with one density threshold, finds the places where collisions are densest, which means sparsely populated areas are under-represented by construction. This is the geographic fairness consideration raised on the [Ethics in Applied Data Science](/data-science-ethics/) page.

**Out-of-Time Validation: 2023 Hotspots Against 2024 Collisions**

Each method flagged ground using 2023 data alone, and the share of the 3,344 collisions in 2024 falling inside that ground was measured. The *collision-affected area*, everywhere within 150 m of any 2023 collision, covers 191.7 km². Because 51.8% of 2024 collisions (48.5% of fatal and serious ones) lie inside it, this is the most that any zone drawn from 2023 data inside that area can capture, and it is why random placement (shown as "lift") is a weak baseline. The DBSCAN Site tier's zone covers 12.1 km², 6.3% of the collision-affected area, so every method was given that much ground. Further budgets of 5% and 10% were also tested.

| Method (12.1 km² flagged) | All 2024 collisions captured (95% interval) | Lift | Fatal/serious captured (95% interval) |
|---|---|---|---|
| **DBSCAN Site (150 m, 5)** | **10.5%** (9.4 to 11.5) | 3.19 | **7.7%** (6.1 to 9.4) |
| GMM, K = 200 | 9.6% (8.6 to 10.6) | 2.92 | 4.0% (2.7 to 5.3) |
| DBSCAN Area (500 m, 10) | 8.6% (7.6 to 9.6) | 2.63 | 6.2% (4.8 to 7.8) |
| Grid 500 m ranked by count | 8.5% (7.5 to 9.4) | 2.63 | 4.6% (3.3 to 6.0) |
| GMM, K = 72 | 7.8% (7.0 to 8.7) | 2.39 | 3.4% (2.2 to 4.7) |
| K-Means, K = 200 | 7.8% (6.9 to 8.7) | 2.37 | 3.3% (2.2 to 4.5) |
| K-Means, K = 72 | 7.1% (6.2 to 8.0) | 2.17 | 3.6% (2.5 to 4.9) |

_Table: capture at equal area. The all-collision columns rank clusters by collision density, and the fatal/serious column ranks them by fatal-or-serious density. The Strict tier (200 m, 10) flags only 2.1% of the area and so is excluded from the comparison; it captures 4.5% (3.8 to 5.2) with the highest lift of any method (4.22)._

![19_capture_all_collisions](19_capture_all_collisions.png)

_Figure 10: Share of 2024 collisions captured against area flagged, ranked by collision density_ 

The DBSCAN Site curve sits at or near the top across the range it covers, with GMM at K = 200 closest. K-Means is lowest, and all methods beat random placement. The Site curve stops at 6.3% because the tier has no more hotspots to flag.

**Differences Between Methods**

Paired bootstrap intervals show which gaps are larger than chance. At the Site-zone area, DBSCAN Site captures more than the grid by **+1.9 percentage points** (95% interval +1.1 to +2.8), K-Means at K = 200 by +2.7 (+1.7 to +3.8), GMM at K = 72 by +2.6 (+1.5 to +3.8) and the Area tier by +1.8 (+1.0 to +2.9). The exception is **GMM at K = 200, where the gap is +0.9 (−0.1 to +2.0) and could be chance**. The 5% budget gives the same pattern: Site captures 8.9% (8.0 to 9.9) against 7.5% (6.6 to 8.4) for the grid (+1.4, +0.5 to +2.3), and again cannot be separated from GMM at K = 200 (8.1%, +0.8, −0.1 to +1.8). The advantage over the grid is real but modest, and a GMM with many components is a serious competitor on collision count.

**Fatal and Serious Collisions**

![20_capture_ksi](20_capture_ksi.png)

_Figure 11: Share of 2024 fatal and serious collisions captured against area flagged, ranked by fatal-or-serious density_ 

This is where DBSCAN Site separates clearly. At the Site-zone area it captures **7.7%** of 2024 fatal and serious collisions (6.1 to 9.4), against 4.6% (3.3 to 6.0) for the grid, 4.0% for GMM at K = 200 and 3.3% for K-Means at K = 200. The gaps are +3.1 points against the grid (+1.4 to +4.9) and +3.7 against GMM at K = 200 (+2.0 to +5.6). Only the Area tier comes close (6.2%), and the gap to Site (+1.5, −0.6 to +3.5) is within chance. A likely reason is that ranking by severity density can order DBSCAN's tight clusters by how severe they are, whereas the large regions of K-Means and GMM dilute severe collisions among many slight ones.

Severity enrichment tells a more cautious story. Captured collisions had a lower fatal-or-serious share than the 2024 county figure of 28.0% in 47 of the 48 method and budget comparisons (the Site tier's share is 73% of the county's, about 20%). The places flagged from 2023 data therefore did not show the county-wide rise in the fatal-or-serious share, which is consistent with hotspots identifying where collisions cluster rather than where they are most severe.

**Stability Across the eps Range**

![21_dbscan_eps_stability](21_dbscan_eps_stability.png)

_Figure 12: DBSCAN capture against area for each eps value, at min_samples of 5 (left) and 10 (right)_ 

The result does not depend on one lucky setting. Site-scale settings (100 to 300 m with `min_samples` of 5) sit at or above the grid benchmark across most of the range, while the 500 m settings fall to or below it. Settings that find few or small hotspots (a small `eps` with a larger `min_samples`) flag little ground, so their curves end early.

**Persistence and Regression to the Mean**

![22_hotspot_persistence](22_hotspot_persistence.png)

_Figure 13: Collisions in each Site hotspot's footprint in 2023 against 2024_ 

The Site zone contained 14.5% of 2023 collisions (566) but only 10.5% of 2024 collisions (350). The ratio, or **retention**, is 0.72 (95% interval 0.64 to 0.81): roughly three-quarters of the apparent concentration persisted, and the remainder is consistent with regression to the mean, the tendency of places chosen for unusually high counts to fall back. Only 22% of the 72 hotspots had at least the 2024 count that falling county volume alone would predict, and three had no collisions at all. The two largest, around grid references TQ 7567 and TQ 7369, fell from 33 to 23 and from 26 to 15, while the hotspot at TQ 5575 rose from 11 to 17. Retention was 0.80 (0.65 to 0.98) for the Strict tier and 0.72 (0.67 to 0.78) for the Area tier, so the intervals overlap and there is no evidence that it depends on scale.

![23_site_zone_capture_map](23_site_zone_capture_map.png)

_Figure 14: 2024 collisions inside (red) and outside (grey) the 2023 Site zone_ 

The 350 collisions captured are concentrated in the same north-western and Medway areas as the 2023 hotspots, with a scatter elsewhere. Most 2024 collisions, 89.5%, fall outside the zone.

**Where the Site Tier Stops**

The Site tier is a short list, not a complete map. At a 10% budget it cannot use the extra ground (it has only 6.3% to give), so others overtake it: GMM at K = 200 captures 13.1% (11.9 to 14.2), the grid 12.4% (11.2 to 13.5), the Area tier 12.1% (11.0 to 13.2) and K-Means at K = 200 11.6%, all above the Site tier's 10.5%. On fatal and serious collisions the Area tier reaches 8.1% (6.5 to 9.9) at that budget, overlapping the Site tier's 7.7%. A larger budget therefore needs a coarser tier or a method that flags more ground, at the cost of precision.

**Replication on 2022 Hotspots Against 2023 Collisions**

The whole procedure, including selecting the Site tier, was repeated one year earlier.

| | 2023 to 2024 | 2022 to 2023 |
|---|---|---|
| Site tier selected | 150 m, 5 | 150 m, 5 |
| Hotspots / collisions inside them | 72 / 13.8% | 78 / 14.6% |
| Site zone | 12.1 km² (6.3%) | 13.3 km² (6.8%) |
| k-distance knee | 1,745 m | 2,380 m |
| Fatal/serious share, fit year to test year | 21.3% to 28.0% | 18.8% to 21.3% |
| All collisions: Site / grid | 10.5% / 8.5% | 11.0% / 8.4% |
| Site minus grid | +1.9 (+1.1 to +2.8) | +2.6 (+1.6 to +3.5) |
| Site minus GMM, K = 200 | +0.9 (−0.1 to +2.0) | +2.3 (+1.4 to +3.2) |
| Fatal/serious: Site / grid | 7.7% / 4.6% | 8.7% / 5.9% |
| Site minus grid | +3.1 (+1.4 to +4.9) | +2.8 (+0.7 to +4.8) |
| Site minus GMM, K = 200 | +3.7 (+2.0 to +5.6) | +5.2 (+3.1 to +7.4) |
| Retention | 0.72 (0.64 to 0.81) | 0.72 (0.64 to 0.82) |

The same setting was chosen, DBSCAN Site came out first in both pairs, and retention was identical, so the result is not specific to one pair of years. The order of the other methods is less stable: K-Means at K = 200 captured 8.8% in the replication, level with GMM at K = 200 (8.7%) and above the grid (8.4%). The one difference is that DBSCAN Site's lead over GMM at K = 200 on all collisions is clearly positive in the replication (+2.3) but not in the primary pair (+0.9), which is why that comparison is best described as small and uncertain. The severity shift also appears in both pairs: serious collisions rose 12.9% from 2022 to 2023 and 13.9% from 2023 to 2024, while slight collisions fell 5.2% and 21.5%.

## Conclusions:

This project set out to show how density-based and model-based clustering can help a county highway authority decide where to concentrate a limited road-safety budget, and when a simpler method is the wrong tool. On that basis the answer is meaningful but bounded. DBSCAN at site scale (150 m, `min_samples` 5) identified 72 small hotspots covering 12.1 km² that captured 10.5% of the following year's collisions and 7.7% of its fatal and serious ones, the highest of any method given the same ground, though not clearly ahead of a 200-component GMM on collision count. The margin over a simple 500 m grid is modest (+1.9 percentage points on all collisions, +3.1 on fatal and serious ones), and the result held in a replication one year earlier.

**A short list, with realistic expectations.** For the decision-maker, the output is a triage list of specific places to investigate, not a forecast. About three-quarters of the apparent concentration persisted into the next year (retention 0.72), so hotspots chosen from a single year should be checked against further years before money is committed. Just as importantly, 89.5% of 2024 collisions fell outside the flagged ground. Hotspot treatment addresses a minority of the problem, and the diffuse remainder needs network-wide measures.

**Method choice should follow the structure of the data.** On the synthetic benchmark each method won where its assumptions held, and K-Means was never the best of the three on any of the four datasets. On the Kent collisions it captured least, or close to least, at equal area (7.1% at K = 72 and 7.8% at K = 200). The Wheat Seeds bridge is a useful counter-example, though: there K-Means (91.90% agreement with the true varieties, as on the [K-Means Clustering](/k-means-clustering/) page) beat a full-covariance GMM and was beaten only by a GMM with one shared covariance (96.19%), while DBSCAN failed because the varieties overlap with no low-density gaps. No method is best in general, and a flexible model can overfit as easily as a rigid one.

**Validation matters more than any single score.** Silhouette ranked methods in the opposite order to the truth on several synthetic datasets, BIC chose the number of Gaussians rather than the number of clusters, and the k-distance knee (1,745 m on Kent) sat more than ten times higher than a site-scale `eps`. The decisive evidence was the out-of-time test, with success defined before modelling began, as the [Data Science Workflow](/data-science-workflow/) page prescribes. It also showed that results depend on scale: the 500 m "Area" tier produced town-sized clusters of up to 2.7 km radius that are areas to study rather than sites to fix.

**The best method depends on what the authority values.** DBSCAN's clearest advantage is on severity, where it captured about 70% more fatal and serious collisions than the grid for the same ground (7.7% against 4.6%). On collision count alone a Gaussian mixture with 200 components was statistically indistinguishable in the primary pair. Beyond the Site tier's 6.3% of the collision-affected area the tier has nothing more to flag, and the grid and the GMM capture more. Whether to prioritise volume or severity, and how far to extend the list, are value judgements that should be made explicitly. This is the accountability point raised on the [Ethics in Applied Data Science](/data-science-ethics/) page.

**Fairness and transparency.** A collision labelled as noise is isolated, not safe. DBSCAN's single density threshold places its hotspots where collisions are densest, so much of the rural south and west of the county has few or none, and a list built this way should be accompanied by a plan for the places it omits. The negative and ambiguous results (the GMM comparison, the Site tier's ceiling, the below-1 severity enrichment) are reported alongside the positive ones for the same reason. DBSCAN's noise label is also different in kind from the anomaly score in the [Anomaly Detection](/anomaly-detection/) project: it falls out of density rather than being tuned against a cost of errors.

**Limitations.** The analysis covers one county and two year pairs, and the data records only police-reported injury collisions. The fatal-or-serious share rose from 18.8% (2022) to 21.3% (2023) and 28.0% (2024), which may reflect changes in recording practice that the data cannot reveal. The analysis uses location only: without traffic volumes or road characteristics, hotspots show where collisions occur, not where the risk per vehicle is highest. The 150 m footprint, the 750 m radius guard and the use of density as the ranking are judgement calls. The bootstrap treats collisions as independent, although several can arise from a single incident, so the intervals may be slightly narrow. There is no formal evaluation by area type, and the synthetic benchmark is small and two-dimensional. As elsewhere in this portfolio, the aim was never business-grade performance. The value lies in the methodology: choosing a clustering method for the structure of the data, and measuring honestly whether it worked.


## Next steps:

With any analysis it is important to assess how the method can evolve to support the business decision and yield tangible benefits. This project is a static analysis of two years of public data, and the extensions below indicate what a real deployment would involve.

**Add exposure and context.** Collision counts say where collisions happen, not where the risk is highest. Joining traffic flow data (for example from the Department for Transport's road traffic statistics), road type and speed limits would allow collisions per vehicle-kilometre to be compared, separating busy but safe roads from quiet but dangerous ones.

**Correct for regression to the mean properly.** Retention of 0.72 shows that single-year hotspots overstate the problem. Road-safety practice uses several years of data and empirical Bayes methods, which shrink each site's count towards what similar sites would be expected to record, to separate lasting hazards from chance clusters. Combining that with the clustering here would give a more defensible ranking.

**Find hotspots across several density levels.** A single `eps` under-serves sparse areas. HDBSCAN or OPTICS, both available in scikit-learn, extract clusters across a range of densities, and a simpler alternative is to run DBSCAN separately on urban and rural roads with different settings. Either would be accompanied by the formal evaluation by area type that this project lacks, to test whether rural and urban hotspots are treated even-handedly.

**Respect the road network.** Collisions happen on roads, but straight-line distance links points on different roads and across junctions. Snapping collisions to a road network and clustering along it, using open road network data, would give hotspots that correspond to road sections an engineer can actually treat.

**Weight by severity and cost.** Ranking by fatal-or-serious density is a coarse version of the weighting a real authority would use. Official cost-of-casualty values could be applied so that rankings reflect the economic case for intervention, with the weighting stated openly.

**From prediction to intervention.** Capturing next year's collisions measures prediction, not whether treating a site would reduce them. A real programme would need before-and-after evaluation of interventions with comparison sites, engineering feasibility, budgets and stakeholder sign-off, none of which this analysis can replicate. In production the annual data release would feed a scheduled pipeline, with monitoring for changes in density and severity mix such as the shift seen here, and a named owner accountable for the list and for what it leaves out.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/dbscan_gmm_final_script.py)
