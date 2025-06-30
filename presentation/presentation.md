---
css: "simple_jk.css"
transition: "slide"
highlightTheme: "github"
slideNumber: false
title: "Presentation Bachelor's Thesis"
---

## Presentation of my Bachelor's Thesis

---

## Contents

* Presentation
    - Topic
    - Objectives
    - Methodology
    - Results
    - Outlook / Future Work
* Discussion
* Conclusion and Feedback

Note:
Present the structure of the 30-minute talk.

---

## Topic

----

This is what I signed up for:

<img src="images/fotofalle_rehwild.jpg" style="width: 50%; margin-bottom: -50px;">

<span style="font-size: 0.4em;">*Image: Bavarian State Institute of Forestry (LWF)*</span>

"Tiere in Fotofallendataset mit KI autmatisch erkennen"

----

This is what it turned into:


<img src="images/example_topic_images.jpg" style="width: 90%; margin-bottom: -50px;">

<span style="font-size: 0.4em;">*Image: Author's own*</span>

"Deep Learning for Biodiversity Monitoring: Automated Classification of Small Mammals Captured in Foto Trap Boxes"

---

## Objectives

----

### These were the core objectives:

- Detect animals in camera trap images  
- Build a preprocessing pipeline  
- Select suitable model architectures  
- Train the classification models  
- Evaluate model performance

---

## Methodology

----
<div style="display: flex; justify-content: space-between; align-items: center; gap: 20px; height: 100%;">

  <div style="flex: 1;">
    <h3>Detection</h3>
    <p>Microsoft MegaDetector was used for an initial detection.</p>
  </div>

  <div style="flex: 1; text-align: right;">
    <img src="images/flow_chart.png" style="width: 100%; max-width: 500px;" />
  </div>

</div>
----
<div style="display: flex; justify-content: space-between; align-items: center; gap: 20px; height: 100%;">

  <div style="flex: 1;">
    <h3>Selecting</h3>
    <img src="images/discarded_img_by_conf.jpg" style="width: 100%; max-width: 500px;" />
  </div>

  <div style="flex: 1; text-align: right;">
    <img src="images/flow_chart.png" style="width: 100%; max-width: 500px;" />
  </div>

</div>
----