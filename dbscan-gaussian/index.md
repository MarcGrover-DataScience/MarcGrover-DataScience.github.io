---

layout: default

title: Road Collisions (DBSCAN and Gaussian Mixture Models)

permalink: /dbscan-gaussian/

---

# This project is in development

## Goals and objectives:

The business objective is to show how density-based and model-based clustering can help a county highway authority decide where to concentrate a limited road-safety budget, and, just as importantly, to show when a simpler method such as K-Means is the wrong tool for the job.

A highway authority has a fixed annual budget for engineering interventions: junction redesign, speed management, signing and lining, anti-skid surfacing. It cannot investigate every location where a collision has been recorded, so it needs a short list of places to examine first. The data available is a year of police-reported injury collisions, each with a map location and a severity (fatal, serious or slight). This project uses the 3,896 collisions recorded in Kent in 2023 to find hotspots, and then asks whether those hotspots still concentrate collisions in 2024 (3,344 collisions), which is when the same authority would actually be spending the budget.

The [K-Means Clustering](/k-means-clustering/) project is currently the only clustering method in this portfolio, and collision locations are exactly the kind of data that stretches its assumptions. K-Means expects compact, roughly spherical clusters, forces every observation into a cluster, and needs K to be chosen in advance. Collisions cluster around junctions and along roads rather than in round blobs, many collisions are isolated events on quiet roads that belong to no hotspot at all, and there is no natural number of hotspots to specify. Two alternatives relax these assumptions in different ways:

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

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/dbscan_gmm_final_script.py)
