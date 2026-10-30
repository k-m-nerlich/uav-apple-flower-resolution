# Studies checked for reported training variability

Non-systematic check referred to in the introduction of the paper ("twelve
recent studies"). Checked on 10 September 2026; selection: recent studies on
flower detection or counting from UAV or ground images that report a detection
or counting accuracy, plus the review [Za26]. For each study we noted whether
variability over repeated training runs and agreement between annotators were
reported.

| Study | Variability over training runs | Annotator agreement |
|---|---|---|
| Zang et al. 2026, IEEE/CAA J. Autom. Sinica (review, 143 references) | none | not measured |
| Qi et al. 2026, Agronomy 16/11, 1103 | announced in the methods ("mean values ± standard deviations from three repeated experiments"), tables report single values only | none |
| Manzoor et al. 2025, Smart Agricultural Technology | none | none |
| Wang et al. 2025, Frontiers in Plant Science (YO-AFD) | none | none |
| Schnalke et al. 2025, Frontiers in Plant Science | none | none |
| Shang et al. 2024, Computers and Electronics in Agriculture | none (variance over phenological stages only) | none |
| Zhang et al. 2023, ISPRS J. Photogramm. Remote Sens. | standard deviation of field measurements only | none |
| Zhang et al. 2022, Biosystems Engineering | leave-one-out CV; variance decomposition of the reference | none |
| Piani et al. 2021, MetroAgriFor | none | none |
| Farjon et al. 2020, Precision Agriculture | none | none |
| Tubau Comas et al. 2019, ISPRS Archives | none | none |
| Low-annotation apple flower counting, 2026 (PubMed Central PMC13316440) | none ("uncertainty" refers to prediction uncertainty for pseudo-labels) | none |

Repeated training as a precedent in the neighbouring field of fruit detection:
Bargoti, S.; Underwood, J. (2017): Deep Fruit Detection in Orchards. ICRA 2017.

The reporting standard proposed in Zang et al. 2026 (absolute and relative
accuracy, latency, deployment cost) includes neither training variability nor
reference reliability. Its "height-accuracy trade-off" is supported
qualitatively for 2/4/8/16 m and quantified at one altitude only; the terms
"GSD" and "ground sampling distance" do not occur in the review.
