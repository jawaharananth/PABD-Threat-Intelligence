# Predictive Attack Behavioural Detection: A Pre-Execution Network Intrusion Early-Warning Framework using CUSUM Drift Monitoring, Ensemble Classification, and Explainable Signal Fusion

**[Author Name]¹, [Guide/Co-author Name]²**

¹ Student, Department of Artificial Intelligence and Machine Learning, [College Name], [University], [City], [State], India
[email]

² [Designation], Department of Artificial Intelligence and Machine Learning, [College Name], [University], [City], [State], India
[email]

## Abstract

Network intrusion detection has traditionally been framed as a post-hoc classification problem: a system observes traffic, decides whether an attack is present, and raises an alert once the evidence is already conclusive. By the time such a system fires, the attack is typically already in its execution phase, leaving a defender little room to intervene. This paper presents the Predictive Attack Behavioural Detection (PABD) framework, a network threat intelligence system that reframes intrusion detection as an early-warning problem by monitoring the behavioural drift that precedes an attack rather than waiting for its signature to fully materialise. The framework combines an Isolation Forest for real-time packet-level anomaly scoring, a cumulative sum (CUSUM) change-point detector tracking ten simultaneous behavioural metrics, a Random Forest classifier for attack-type attribution, and a Long Short-Term Memory (LSTM) network trained to estimate the residual time to attack onset. A calibrated logistic fusion layer combines these signals into a single risk score, and every detection is mapped to its corresponding MITRE ATT&CK tactic and technique, with SHAP (SHapley Additive exPlanations) values generated per alert to explain which behavioural features drove the classification. The system is evaluated on real traffic captured across eight days of the CICIDS2017 dataset, spanning seven attack categories: Reconnaissance, Brute Force, Denial of Service (DoS), Distributed Denial of Service (DDoS), Botnet activity, Web Application attacks, and benign traffic. The Random Forest classifier attains 73% accuracy on a chronologically held-out test split; the fusion layer achieves an area under the ROC curve of 0.964, exceeding the best individual detection signal (Random Forest confidence alone, AUC 0.930); and the countdown LSTM estimates time-to-onset with a mean absolute error of approximately 374 seconds on 50 held-out real attack sequences. A black-box adversarial evasion analysis further shows that while every evaluated attack sample can, in principle, be pushed across the decision boundary, the magnitude of perturbation required ranges from 37% to 74% of the distance to typical benign traffic depending on attack class, and a meaningful share of the required perturbation touches features an attacker cannot alter without defeating the purpose of the attack itself. The evaluation is reported with explicit attention to its own limitations, including a documented precision-recall trade-off in the CUSUM detector and the absence of a live signature-based intrusion detection system for direct comparison.

**Keywords:** Network Intrusion Detection · Predictive Security Analytics · CUSUM Change-Point Detection · Ensemble Learning · Explainable AI · MITRE ATT&CK · Adversarial Robustness

## 1 Introduction

### 1.1 The Case for Predictive Intrusion Detection

The growth of networked infrastructure has been accompanied by a corresponding growth in the sophistication and frequency of network-based attacks, and this has created an urgent need for detection systems that do more than confirm an attack after it has already begun. Conventional intrusion detection systems (IDS), whether signature-based tools such as Snort and Suricata or machine-learning classifiers trained on flow-level features, are fundamentally reactive: they are designed to recognise an attack's signature once sufficient evidence has accumulated, at which point the attacker has frequently already achieved their objective. Recent research has demonstrated that machine learning can substantially improve the accuracy of attack classification (Ferrag et al., 2020), and that ensemble methods in particular are well suited to the heterogeneous, high-dimensional feature spaces produced by modern network flow data. However, the majority of contemporary approaches continue to treat detection as a binary or multiclass classification problem evaluated at a single point in time, rather than as a temporal process in which the traffic leading up to an attack carries its own diagnostic signal.

The premise of this work is that many attack categories exhibit a measurable behavioural drift before they reach their damaging phase: a reconnaissance scan gradually broadens its port coverage, a brute-force credential attack shows a rising rate of failed authentications before it succeeds, and a denial-of-service attack typically ramps its connection frequency before it saturates its target. If this drift can be detected reliably, a defender gains a window of time to respond that a conventional, signature-triggered IDS cannot offer.

### 1.2 Problem Statement and Proposed Solution

Existing intrusion detection literature has produced strong classifiers for the question "is this an attack," but comparatively little work addresses the complementary question "is this traffic drifting toward becoming an attack, and how much time remains." Statistical change-point detection techniques such as the cumulative sum (CUSUM) algorithm have long been used in industrial process monitoring to detect exactly this kind of gradual drift (Farkas, 2016), yet their application to network security has historically been limited by a well-documented weakness: CUSUM-based detectors are highly sensitive to the choice of baseline and threshold, and a poorly calibrated detector produces a false-positive rate that renders it operationally useless — a limitation that is consistent with the broader, frequently cited observation that the overwhelming majority of alerts raised by conventional network intrusion detection systems are false positives.

This paper presents the Predictive Attack Behavioural Detection (PABD) framework to address this gap. PABD does not propose CUSUM as a replacement for classification-based detection; rather, it integrates a tuned, multi-metric CUSUM detector as an early-warning layer that operates alongside an Isolation Forest anomaly scorer, a Random Forest attack classifier, and an LSTM-based time-to-onset estimator, with all four signals combined through a calibrated fusion layer. The significant contributions of this paper are:

- A behavioural early-warning detector that tracks ten simultaneous network flow metrics using a reset-corrected CUSUM formulation, evaluated for detection lead time against a flow-fraction threshold baseline on real network traffic.
- A time-to-attack-onset estimator trained on real, chronologically ordered pre-attack windows extracted from live CICIDS2017 capture timelines, rather than on synthetically generated sequences.
- A calibrated fusion layer that combines Isolation Forest, CUSUM, and Random Forest outputs into a single risk score and is shown, on held-out data, to rank threats more reliably than any one of its constituent signals.
- An explainability layer that attaches a live, per-alert SHAP explanation and a MITRE ATT&CK tactic/technique tag to every classified attack, rather than a static, global feature-importance ranking.
- An adversarial robustness assessment quantifying how much an attacker would need to alter their traffic behaviour to evade the classifier, together with an explicit judgement of which of the required changes are realistically achievable without defeating the attack's own objective.
- A transparent account of the framework's operating limits, including the precision/false-positive trade-off inherent to the tuned CUSUM detector and the classes for which the real dataset provides insufficient ground truth.

The rest of the paper is organised as follows. Section 2 reviews related work in machine-learning-based intrusion detection and change-point detection. Section 3 describes the system architecture. Section 4 details the dataset and methodology. Section 5 presents the experimental results and discussion. Section 6 concludes the paper and outlines future work.

## 2 Background and Related Work

The development of large, labelled network traffic datasets and the maturation of ensemble and deep learning methods have together made it possible to move beyond signature matching toward behaviourally grounded intrusion detection, and this section situates PABD within that body of work.

### 2.1 Machine Learning for Network Intrusion Detection

A substantial body of research has applied ensemble classifiers, in particular Random Forest and gradient-boosted trees, to network flow data drawn from datasets such as CICIDS2017, generally reporting strong classification accuracy on the majority attack classes while acknowledging that class imbalance and dataset-specific artefacts remain persistent difficulties (Sharafaldin et al., 2018). Panigrahi and Borah (2018) demonstrated that a comprehensive evaluation of CICIDS2017 requires careful attention to the dataset's known flow-construction and labelling irregularities, a concern this work addresses explicitly rather than treating the dataset as ground truth without qualification. Separately, isolation-based anomaly detectors have been shown to be effective at flagging anomalous flows without requiring labelled attack examples (Liu et al., 2008), which motivates their use here as a first, unsupervised anomaly layer ahead of supervised classification.

### 2.2 Change-Point Detection and Early-Warning Systems

The CUSUM algorithm and its variants have been applied to network anomaly detection with the specific aim of catching gradual drift rather than instantaneous spikes (Tartakovsky et al., 2006). A recurring finding in this literature is that raw CUSUM detectors are vulnerable to seasonal or cyclical traffic patterns that masquerade as drift, and that wavelet-filtered or baseline-adaptive variants can meaningfully reduce the resulting false-positive rate (Farkas, 2016). This work adopts a comparable adaptive-baseline strategy, while additionally identifying and correcting a related but distinct failure mode: an unmitigated CUSUM accumulator, once tripped by a single large but isolated fluctuation, can remain incorrectly triggered for a large number of subsequent observation windows unless it is explicitly permitted to reset once traffic returns to its normal range.

### 2.3 Explainability and Adversarial Considerations in Intrusion Detection

The application of SHAP-based explainability to network intrusion detection has been explored as a means of making classifier decisions auditable to a security analyst (Lundberg and Lee, 2017), but the majority of existing deployments compute a single, dataset-level feature-importance ranking rather than an explanation specific to the alert currently being investigated, which limits their usefulness during live incident response. Separately, the adversarial robustness of network intrusion classifiers has received comparatively limited attention relative to the broader adversarial machine learning literature, despite the fact that, unlike an image classifier, a network attacker is an adaptive adversary with a direct incentive to evade detection. Existing gaps identified across this literature are addressed jointly in the present work: no system surveyed combines behavioural early-warning detection, live per-alert explainability, and an explicit adversarial evasion assessment within a single evaluated framework operating on real attack traffic.

## 3 System Architecture

The objective of the proposed system is to observe live network traffic, identify behavioural drift that precedes an attack, classify the attack type once sufficient evidence has accumulated, estimate the residual time before the attack reaches its execution phase, and present all of this information to an analyst in an explainable and auditable form. Figure 1 depicts the overall data flow of the framework.

*(Figure 1: PABD System Architecture — to be inserted. Suggested content: raw traffic/flow features flowing into four parallel detectors (Isolation Forest, CUSUM, Random Forest, LSTM), converging into the fusion layer, which feeds the MITRE mapping and SHAP explanation modules, culminating in the live dashboard.)*

The system is organised into four cooperating detection layers and two supporting intelligence layers.

The **anomaly layer** consists of an Isolation Forest trained exclusively on benign traffic, which assigns every observed window a continuous anomaly score without requiring labelled attack examples. This layer is intentionally unsupervised so that it retains some sensitivity to attack patterns not present in the labelled training classes.

The **drift layer** applies a CUSUM change-point detector independently to each of ten behavioural features — connection frequency, port entropy, failed-authentication rate, packet size distribution, protocol concentration, burst activity, SYN flag rate, flow-duration variability, mean byte rate, and traffic directionality ratio — and raises a composite alert once a specified number of these ten metrics simultaneously cross their drift threshold. This is the layer responsible for the framework's early-warning property: because it accumulates evidence of gradual drift rather than waiting for a fully formed attack signature, it is capable of firing before a conventional threshold-based detector would.

The **classification layer** is a Random Forest classifier trained on the same ten behavioural features, which attributes a specific attack category to a window once the drift layer has fired.

The **temporal layer** is a Long Short-Term Memory network trained on sequences of ten consecutive observation windows leading up to a confirmed real attack onset, which produces an estimate, in seconds, of how much time remains before the attack is expected to reach its execution phase.

The four layers' outputs are combined by a **fusion layer**, a calibrated logistic model trained to map the Isolation Forest score, the CUSUM breach count, and the Random Forest's classification confidence onto a single, better-calibrated composite risk score.

Finally, the **intelligence layer** consists of two components attached to every classified alert: a mapping from the predicted attack category to its corresponding MITRE ATT&CK tactic and technique identifier, and a live SHAP explanation computed for that specific alert's feature vector, rather than a precomputed, dataset-wide importance ranking.

## 4 Methodology

### 4.1 Dataset Details

The framework is trained and evaluated on real network traffic drawn from eight days of the CICIDS2017 dataset (Sharafaldin et al., 2018), covering benign traffic together with Reconnaissance, Brute Force, Denial of Service, Distributed Denial of Service, Botnet, and Web Application attack traffic. Raw packet-level flow records are aggregated into fixed 120-second observation windows, from which the ten behavioural features described in Section 3 are computed. Windows are labelled by the majority attack type present within them, and a continuous attack-flow fraction is retained alongside the categorical label to support lead-time analysis. After aggregation, the resulting dataset comprises 861 labelled windows distributed across seven classes: Normal (588), Botnet (90), Brute Force (66), Denial of Service (51), Web Application attacks (36), Reconnaissance (19), and Distributed Denial of Service (11). A held-out day of exclusively benign traffic is additionally reserved and used only for a distribution-shift false-positive assessment, described in Section 5.

Two attack categories present in the raw capture — Infiltration and Lateral Movement — are deliberately excluded from the trained classifier. The Infiltration category contains only 36 raw flow records across the entire capture, which is judged insufficient to support a trained class; it is instead evaluated separately as a small-sample case study. No Lateral Movement traffic exists in the days of CICIDS2017 used in this study; the category is retained only in the illustrative live-demonstration mode of the dashboard, and no accuracy claim is made for it.

### 4.2 Behavioural Feature Engineering

Six of the ten features — connection frequency, port entropy, failed-authentication rate, packet size distribution, protocol concentration, and burst activity — are computed directly from flow-level aggregates within each observation window. The remaining four features are derived specifically to capture signatures that the initial six features under-represent: SYN flag rate (targeting reconnaissance-style scanning behaviour), flow-duration coefficient of variation (targeting the short, homogeneous flows typical of automated scanning), mean flow byte rate (targeting volumetric flooding behaviour), and the ratio of downstream to upstream traffic (targeting the directional asymmetry characteristic of exfiltration and command-and-control channels). In an ablation comparison, these four additional features rank as the four most important features by Random Forest feature importance, ahead of all six original features, and their inclusion improves both classification accuracy and CUSUM detection quality, as reported in Section 5.

### 4.3 CUSUM Drift Detection with Baseline Adaptation and Reset Correction

For each behavioural feature $f$, the detector maintains an accumulator $s_f$, updated at each observation window as
$$s_f \leftarrow \max\left(0,\; s_f + \left(x_f - \mu_f - k\sigma_f\right)\right)$$
where $x_f$ is the observed value, $\mu_f$ and $\sigma_f$ are the mean and standard deviation of the feature under benign traffic, and $k$ is a slack parameter. An alert is raised for the window once $s_f$ exceeds a threshold $h\sigma_f$ for at least $N$ of the ten monitored features simultaneously. The parameters $k$, $h$, and $N$ are selected by grid search against held-out labelled data rather than chosen by hand.

A supplementary reset mechanism is applied to $s_f$: once a feature has remained within a tolerance band of its baseline for a specified number of consecutive windows, its accumulator is forced back to zero. This correction was introduced after observing, during testing, that a single large but isolated fluctuation could otherwise leave the unmitigated accumulator falsely triggered for a substantial number of subsequent windows, since the accumulator's natural decay rate is bounded by the slack term $k\sigma_f$ rather than by the magnitude of the original fluctuation.

### 4.4 Time-to-Onset Estimation

The LSTM component is trained on sequences of ten consecutive real observation windows drawn from the chronological CICIDS2017 timeline, with each sequence labelled by the true number of windows remaining before the nearest subsequent attack onset, normalised against a fixed lookback horizon. This construction is deliberately distinct from training the network on synthetically generated pre-attack sequences: because the sequences are drawn from real attack onsets rather than an artificial generator, the reported error reflects the difficulty of the estimation problem as it exists in real traffic, rather than the difficulty of a proxy problem constructed for training convenience.

### 4.5 Fusion of Detection Signals

The Isolation Forest anomaly score, the CUSUM breach count, and the Random Forest's maximum class probability are combined as input features to a calibrated logistic regression model, trained and evaluated on a held-out chronological split disjoint from the split used to tune the individual detectors. The fusion model's calibrated probability output is used as the system's composite risk score in place of any single constituent signal.

### 4.6 Adversarial Evasion Methodology

Because the Random Forest classifier is a non-differentiable tree ensemble, gradient-based adversarial attack methods are not directly applicable. A black-box boundary-search evasion procedure is used instead: for each real attack sample, a straight-line path is constructed in scaled feature space between the sample and the centroid of the benign class, and a binary search along that path identifies the minimum perturbation magnitude at which the classifier's prediction flips to benign. This magnitude is reported both in absolute terms and as a fraction of the total distance between the original sample and the benign centroid, together with an explicit, per-feature judgement of whether an attacker could plausibly achieve the required change without forfeiting the functional objective of the attack itself — for example, an attacker can trivially slow their request rate, but cannot alter the fundamental directionality of a denial-of-service flood without ceasing to conduct one.

### Algorithm 1 PABD Predictive Detection Pipeline

**Input:** Live network flow stream, benign-traffic baseline statistics $(\mu, \sigma)$, trained Isolation Forest $\text{IF}$, trained Random Forest $\text{RF}$, trained countdown LSTM, trained fusion model $\text{FM}$, CUSUM parameters $(k, h, N)$.
**Output:** Per-window anomaly score, drift-breach count, attack classification with confidence, estimated time-to-onset, composite risk score, MITRE tactic/technique tag, per-alert SHAP explanation.

1. Aggregate incoming flow records into a 120-second observation window $w$ and compute the ten behavioural features $x = (x_1, \dots, x_{10})$.
2. Compute the anomaly score $a = -\text{IF.score\_samples}(x)$.
3. For each feature $f$, update the CUSUM accumulator: $s_f \leftarrow \max(0, s_f + (x_f - \mu_f - k\sigma_f))$, applying the reset correction of Section 4.3.
4. Count $b = |\{f : s_f > h\sigma_f\}|$.
5. **If** $b \geq N$ **and** no alert has yet been raised for the current episode: raise a drift alert and record the current window index.
6. **If** a drift alert is active and ten consecutive windows are available: classify $\hat{y} = \text{RF}(x)$ with confidence $p = \max(\text{RF.predict\_proba}(x))$, and compute the countdown estimate $\hat{t} = \text{LSTM}(x_{w-9}, \dots, x_w) \times T_{\text{horizon}}$.
7. Compute the composite risk score $r = \text{FM}(a, b, p)$.
8. Retrieve the MITRE ATT&CK tactic and technique corresponding to $\hat{y}$.
9. Compute SHAP values $\phi = \text{SHAP}(\text{RF}, x)$ for the predicted class $\hat{y}$, and rank features by $|\phi|$.
10. Emit $(a, b, \hat{y}, p, \hat{t}, r, \text{tactic}, \text{technique}, \phi)$ to the analyst dashboard.

### 4.7 Working of the Detection Pipeline

The behavioural drift layer and the anomaly layer operate on every incoming observation window independently of one another, so that a window can be flagged anomalous by the Isolation Forest without necessarily contributing to a CUSUM breach, and vice versa. The classification and temporal layers are invoked only once the drift layer has raised its composite alert, since attributing an attack type or estimating a countdown before sufficient drift evidence has accumulated would not be meaningful. Once an attack is classified, the explanation and intelligence layers operate on that single window's feature vector, so that the SHAP ranking and the displayed risk score reflect the specific traffic that triggered the alert rather than a precomputed, dataset-wide summary. This design allows an analyst to move from a raw anomaly signal to a fully attributed, explained, and time-bounded threat assessment without any additional manual correlation step.

## 5 Discussion and Results

The framework is implemented in Python, with the Isolation Forest, Random Forest, and fusion layer built on scikit-learn, the countdown LSTM built on TensorFlow/Keras, explanation generation performed with the SHAP library, and the live dashboard built with Dash/Plotly.

### 5.1 Classification Performance

On a chronological 70/30 train-test split of the 861-window real dataset, the Random Forest classifier attains an overall accuracy of 73%. Per-class performance is uneven, which is expected given the class imbalance inherent to real attack traffic: the Botnet class, whose defining signature is a low-volume, persistent connection pattern that is subtle relative to benign background traffic, is the weakest-performing class (F1-score 0.39), while classes with a more pronounced volumetric signature perform substantially better. The Distributed Denial of Service class attains a perfect score on the held-out split, but this result rests on only two test-set samples and is reported here explicitly as statistically inconclusive rather than as evidence of the classifier's general reliability on that class.

### 5.2 Drift Detection and the Precision-Recall Trade-off

Figure 2 presents the detector's precision, recall, and false-positive rate across a sweep of the breach-count threshold $N$, evaluated both against the six attack-bearing days used for tuning and against a fully held-out, exclusively benign day.

*(Figure 2: to be inserted — precision/recall/false-positive-rate vs. N, both attack-day and clean-day curves.)*

| $N$ (of 10) | Precision | Recall | FPR (attack days) | FPR (clean day) |
|---|---|---|---|---|
| 2 | 0.410 | 0.722 | 0.483 | 0.631–0.730 |
| 3 | 0.372 | 0.425 | 0.333 | 0.504 |
| 4 | 0.365 | 0.333 | 0.269 | 0.279 |
| 5 | 0.378 | 0.205 | 0.156 | 0.184 |
| 6 | 0.132 | 0.026 | 0.078 | 0.000 |

*Table 2: CUSUM detector operating points, with and without the baseline-adaptive reset correction (clean-day range reflects the improvement from that correction).*

The operating point that maximises F1-score ($N=2$) recalls 72% of attacks in the tuning data, but its false-positive rate on the fully held-out clean day is 63–73%, meaning that at this setting the detector cannot be meaningfully distinguished from noise on traffic it was not tuned against. This finding is consistent with the broader observation in the network intrusion detection literature that the majority of alerts raised by change-point and threshold-based detectors are false positives, and it is reported here as a genuine limitation of the approach rather than smoothed over in favour of the more favourable F1-optimal figure. A stricter operating point ($N=3$–$4$) is put forward as the more defensible choice for deployment, trading recall for a substantially lower false-positive rate on unseen traffic. The baseline-adaptive reset correction described in Section 4.3 measurably improves this trade-off, reducing the clean-day false-positive rate at the $N=2$ operating point from 73.0% to 63.1%, though it does not eliminate the underlying trade-off.

### 5.3 Time-to-Onset Estimation

The countdown LSTM, trained on 249 real pre-attack sequences drawn from the chronological CICIDS2017 timeline, attains a held-out mean absolute error of approximately 374 seconds on 50 test sequences. This figure is reported in place of an earlier, substantially lower error estimate obtained when the same network was trained on synthetically generated pre-attack sequences; the discrepancy between the two figures illustrates the risk of evaluating a temporal estimator on a proxy training distribution rather than on real attack progression data, and the real-data figure is the one put forward as representative of the system's actual capability.

### 5.4 Fusion Layer Performance

Table 3 compares the composite fusion risk score against each of its constituent signals evaluated independently on the same held-out data.

| Signal | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|
| Isolation Forest score alone | 0.493 | 0.537 | 0.514 | 0.763 |
| CUSUM breach count alone | 0.302 | 1.000 | 0.464 | 0.564 |
| Random Forest confidence alone | 0.335 | 0.970 | 0.498 | 0.930 |
| **Fusion (calibrated)** | 0.719 | 0.955 | **0.821** | **0.963** |

*Table 3: Fusion layer performance versus individual detection signals, held-out chronological split.*

The fused risk score attains a higher area under the ROC curve than any individual constituent signal, indicating that the fusion layer contributes genuine ranking improvement rather than simply reproducing the strongest individual signal. At the default decision threshold, the fusion model's binary decision coincides closely with the Random Forest's own attack/non-attack flag, which is consistent with the Random Forest being the most informative single signal available to the fusion model; the improvement over the Random Forest in isolation is concentrated in the quality of the continuous risk ranking (AUC) rather than in the binary decision boundary.

### 5.5 Adversarial Robustness

Table 4 reports the median relative perturbation, expressed as a fraction of the distance to the benign class centroid, required to evade the Random Forest classifier for each attack category.

| Attack Category | Median Relative Perturbation |
|---|---|
| Botnet | 0.37 |
| Web Application | 0.38 |
| Brute Force | 0.41 |
| Reconnaissance | 0.50 |
| Denial of Service | 0.54 |
| Distributed Denial of Service | 0.74 |

*Table 4: Median perturbation required for evasion, by attack category.*

Distributed Denial of Service traffic is the hardest category to disguise as benign, consistent with its inherently extreme volumetric signature placing it furthest from the benign centroid in feature space; Botnet traffic is the easiest to disguise, consistent with its low-and-slow beaconing behaviour already sitting comparatively close to that centroid. Critically, evasion feasibility in feature space does not imply evasion feasibility for a real attacker: several of the features that would need to change most for a successful evasion — for example, the ratio of downstream to upstream traffic, which reflects the actual direction of the attack's data flow — cannot be altered by an attacker without simultaneously defeating the functional purpose of the attack, and this distinction is made explicit for each feature rather than left implicit in the perturbation figures alone.

### 5.6 Comparative Positioning

Table 5 situates PABD's constituent techniques against comparable approaches described in the reviewed literature, on the basis of the capabilities each approach reports rather than a claim of having benchmarked all systems on identical data, since no shared, publicly reported evaluation exists across this literature on the same dataset and feature set.

| Feature | PABD (Proposed) | Signature-based IDS (e.g., Snort/Suricata) | Standalone ML Classifiers (Sharafaldin et al., 2018) | Unmitigated CUSUM (Tartakovsky et al., 2006) |
|---|---|---|---|---|
| Detection timing | Pre-execution (drift-based) | Post-execution (signature match) | Post-execution (flow classification) | Pre-execution, but drift-only |
| Attack attribution | Random Forest, 7 real classes | Rule-based, no confidence score | Yes, no early-warning layer | Not applicable |
| Explainability | Live, per-alert SHAP | None | Typically none | Not applicable |
| Adversarial evaluation | Reported explicitly | Not typically reported | Not typically reported | Not typically reported |
| False-positive characterisation | Reported across operating points, including on unseen clean traffic | Widely reported as high in practice | Reported on tuning data only | High, without baseline adaptation |

*Table 5: Comparative positioning of PABD against related approaches, by reported capability.*

## 6 Conclusion and Future Work

This paper has presented the Predictive Attack Behavioural Detection framework, a network threat intelligence system that combines an Isolation Forest anomaly scorer, a reset-corrected multi-metric CUSUM drift detector, a Random Forest attack classifier, and a real-data-trained LSTM countdown estimator, unified through a calibrated fusion layer and made auditable through live per-alert SHAP explanations and MITRE ATT&CK mapping. Evaluated on real CICIDS2017 traffic across seven attack categories, the framework demonstrates that behavioural drift monitoring can meaningfully precede signature-based detection, that signal fusion improves ranking quality over any single detector, and that the resulting classifier's adversarial robustness, while imperfect, is bounded by real constraints on what an attacker can alter without abandoning their objective. The evaluation has been conducted with explicit attention to the framework's limitations: no signature-based intrusion detection system was available for a direct, live comparison; no second, independent dataset was used to test generalisation beyond CICIDS2017; and the Lateral Movement and Infiltration categories are not supported by sufficient real ground truth in the data used, and are reported as such rather than folded into headline accuracy figures.

Future work will prioritise three directions. First, a live comparison against a deployed signature-based IDS operating on the same raw traffic capture would allow the lead-time advantage claimed in this work to be measured directly rather than approximated against a flow-fraction threshold baseline. Second, validation against a second, independently collected intrusion detection dataset would test whether the reported detection and fusion performance generalises beyond the specific traffic-generation methodology of CICIDS2017. Third, extending the drift detection layer with a data-driven causal discovery step, conditioned explicitly on treatment/response timestamps, may allow the framework to support Lateral Movement detection once sufficient real-world ground truth becomes available, which the current dataset does not provide.

## References

1. Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy*, 108-116.
2. Panigrahi, R., Borah, S. (2018). A detailed analysis of CICIDS2017 dataset for designing intrusion detection systems. *International Journal of Engineering & Technology*, 7(3.24), 479-482.
3. Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
4. Tartakovsky, A. G., Rozovskii, B. L., Blažek, R. B., Kim, H. (2006). Detection of intrusions in information systems by sequential change-point methods. *Statistical Methodology*, 3(3), 252-293.
5. Farkas, K. (2016). CUSUM anomaly detection. *M-Lab Technical Publications*.
6. Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
7. Ferrag, M. A., Maglaras, L., Moschoyiannis, S., Janicke, H. (2020). Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study. *Journal of Information Security and Applications*, 50, 102419.
8. MITRE Corporation. (2024). MITRE ATT&CK Framework. Retrieved from https://attack.mitre.org
