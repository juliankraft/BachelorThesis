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
* Wrap up and Feedback

Note:
Present the structure of the 30-minute talk.

---

## Topic

----

This is what I signed up for:

<img src="images/fotofalle_rehwild.jpg" style="width: 50%; margin-bottom: -30px;"><br>
<span style="font-size: 0.4em;">*Image: Bavarian State Institute of Forestry (LWF)*</span>

"Tiere in Fotofallendataset mit KI autmatisch erkennen"

----

This is what it turned into:


<img src="images/example_topic_images.jpg" style="width: 90%; margin-bottom: -40px;"><br>
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

### Processing of a Sequence

----
<img src="images/flow_chart.png" style="width: 48%; margin-bottom: -30px;"><br>
<span style="font-size: 0.4em;">*Figure: Author's own*</span>
----
<div style="display: flex; justify-content: space-between; align-items: center; gap: 20px; height: 100%;">

<div style="flex: 1;">
<h3>Selecting</h3>
<img src="images/discarded_img_by_conf.jpg" style="width: 100%; max-width: 500px; margin-bottom: -30px;"><br>
<span style="font-size: 0.4em;"><em>Figure: Author's own</em></span>
</div>

  <div style="flex: 1; text-align: right;">
    <img src="images/flow_chart.png" style="width: 80%; max-width: 500px;" />
  </div>

</div>
----
<img src="images/flow_chart.png" style="width: 48%; margin-bottom: -30px;"><br>
<span style="font-size: 0.4em;">*Figure: Author's own*</span>
---

## Methodology

### Training

----

!!!! slides missing !!!!

---

## Methodology

### Evaluation

----

!!!! slides missing !!!!

---

## Results

----

### Comparing Different Model Architectures

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 40px;">

  <!-- Table column -->
  <div style="flex: 1; text-align: center;">
    <div style="font-size: 0.4em; margin-bottom: -15px;"><em>Table: BalAcc of all models – mean ± standard deviation; best values highlighted.</em></div>
    <img src="images/table_compare.png" style="width: 100%; max-width: 500px;">
  </div>

  <!-- Bar chart column -->
  <div style="flex: 1; text-align: center;">
    <img src="images/bal_acc_img.jpg" style="width: 100%; max-width: 500px; margin-bottom: -15px;">
    <div style="font-size: 0.4em; margin-top: 0px;"><em>Figure: Author's own</em></div>
  </div>

</div>

----

### Pretrained EfficientNet-B0

<span style="font-size: 0.4em;">*Table: Class-wise precision, recall, F1-score, and support for the pretrained EfficientNet-B0.*</span>
<img src="images/table_best.png" style="width: 60%; margin-top: 0px;">

---

## Discussion

----