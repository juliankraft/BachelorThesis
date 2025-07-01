---
css: "simple_jk.css"
transition: "slide"
highlightTheme: "github"
slideNumber: false
title: "Presentation Bachelor's Thesis"
---

# Presentation Bachelor's Thesis

---

## Contents

----

* Presentation  
    - Topic  
    - Objectives  
    - Methodology  
    - Results  
    - Outlook 
* Discussion  
* Wrap up and Feedback

Note:
Present the structure of the 30-minute talk.

---

## Topic

----

### What I signed up for
"Tiere in Fotofallendataset mit KI autmatisch erkennen"
<div class="image-block">
  <img src="images/fotofalle_rehwild.jpg" style="width: 50%;">
  <div class="figure-caption">Image: Bavarian State Institute of Forestry (LWF)</div>
</div>

----

### What it turned into
"Deep Learning for Biodiversity Monitoring: Automated Classification of Small Mammals Captured in Foto Trap Boxes"
<div class="image-block">
  <img src="images/example_topic_images.jpg" style="width: 90%;">
  <div class="figure-caption">Image: Author's own</div>
</div>

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

<div class="image-block">
  <img src="images/flow_chart.png" style="width: 48%;">
  <div class="figure-caption">Figure: Author's own</div>
</div>

----

<div class="image-row">

  <div class="image-col">
    <img src="images/discarded_img_by_conf.jpg">
    <div class="figure-caption">Figure: Author's own</div>
  </div>

  <div class="image-col">
    <img src="images/flow_chart.png" style="width: 80%;">
    <div class="figure-caption">Figure: Author's own</div>
  </div>

</div>

----

<div class="image-block">
  <img src="images/flow_chart.png" style="width: 48%;">
  <div class="figure-caption">Figure: Author's own</div>
</div>

---

## Methodology

### Training

----

_**Slides missing – Add content here.**_

---

## Methodology

### Evaluation

----

_**Slides missing – Add content here.**_

---

## Results

----

### Comparing Different Model Architectures

<div class="image-row">

  <div class="image-col">
    <div class="table-caption">Table: BalAcc of all models – mean ± standard deviation; best values highlighted.</div>
    <img src="images/table_compare.png">
  </div>

  <div class="image-col">
    <img src="images/bal_acc_img.jpg">
    <div class="figure-caption">Figure: BalAcc</div>
  </div>

</div>

----

### Pretrained EfficientNet-B0

<div class="image-row">

  <div class="image-col">
    <div class="table-caption">Table: Class-wise precision, recall, F1-score, and support for the pretrained EfficientNet-B0.</div>
    <img src="images/table_best.png">
  </div>

  <div class="image-col">
    <img src="images/conf_matrix_best.jpg" style="width: 80%;">
    <div class="figure-caption">Figure: Confusion Matrix</div>
  </div>

</div>

----

### Stoats hard to detect – easy to classify

<div class="image-block">
  <img src="images/mustela_hard_detect.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Not detected Stoats</div>
</div>

----

### Stoats hard to detect – easy to classify

<div class="image-block">
  <img src="images/mustela_easy_classify.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Stoats - Easy to classify</div>
</div>

----

### Looking into some errors

<div class="image-block">
  <img src="images/false_class_snails.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Detected Snails – Classified as Mammals</div>
</div>

---

## Outlook

----

_**Slides missing – Add content here.**_

---

## Discussion

----

_Add your discussion points here._

---

## Wrap up and Feedback

----

_Add your Wrap Up._
