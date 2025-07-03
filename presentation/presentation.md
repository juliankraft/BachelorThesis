---
css: "simple_jk.css"
transition: "slide"
highlightTheme: "github"
slideNumber: false
title: "Presentation Bachelor's Thesis"

---
## Deep Learning for Biodiversity Monitoring

Note:
Titel der Arbeit

---
## Outline

----
  <div class="image-col">
    <img src="images/outline.svg" style="width: 1000%;">
  </div>

Note:
Present the structure of the 30-minute talk.
* Presentation  
  - Topic  
  - Objectives  
  - Methodology  
  - Results  
  - Outlook 
* Discussion  
* Feedback

---
## Topic

----
### What I signed up for
"Tiere in Fotofallendataset mit KI automatisch erkennen"
<div class="image-block">
  <img src="images/fotofalle_rehwild.jpg" style="width: 50%;">
  <div class="figure-caption">Image: Bavarian State Institute of Forestry (LWF)</div>
</div>

----
### What it turned into
"Deep Learning for Biodiversity Monitoring: Automated Classification of Small Mammals Captured in Foto Trap Boxes"
<div class="image-block">
  <img src="images/example_topic_images.jpg" style="width: 90%;">
  <div class="figure-caption">Figure: Author's own example</div>
</div>

Note:
already way in the core of the topic

----
### Biodiversity

<div class="image-block">
  <img src="images/biodiversity_loss.png" style="max-width: 500px;">
  <div class="figure-caption">
    Figure: Global trends in biodiversity loss (Brondízio et al., 2019, IPBES)
  </div>
</div>

<div style="max-width: 80%; margin: 40px auto 0; font-size: 0.3em;">
  Source: Brondízio, E. S., Settele, J., Díaz, S., & Ngo, H. T. (Eds.). 
  (2019). 
  <i>The Global Assessment Report of the Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services</i>. 
  IPBES, Bonn.
</div>

Note:
This figure illustrates long-term declines across vertebrate groups.

----
### Small Mammals

ecologically important and often overlooked
<div class="image-block">
  <img src="images/example_mammal.jpg" style="max-width: 500px;">
  <div class="figure-caption">Image: From the Mammalia Dataset</div>
</div>

----
### Wildlife@Campus Project

<div class="image-block">
  <img src="images/campus.svg" style="max-width: 500px;">
  <div class="figure-caption">Figure: Author's own</div>
</div>


----
### Why Automation Matters

📷 **Millions of Images**

⚙️ **Automation is Key**  

<div class="image-block">
  <img src="images/illustration_software.png" style="max-width: 500px;">
  <div class="figure-caption">Image: Illustration by ChatGPT 4o</div>
</div>

---
## Objectives

----
### These were the core objectives:

  <div class="image-col">
    <img src="images/objectives.svg" style="width: 1000%;">
  </div>

---
## Methodology

----
<div class="image-block">
  <img src="images/flow_chart.svg" style="width: 48%;">
</div>

----
<div class="image-row">

  <div class="image-col">
    <img src="images/discarded_img_by_conf.jpg">
    <div class="figure-caption">Figure: Percentage of images discarded</div>
  </div>

  <div class="image-col">
    <img src="images/flow_chart.svg" style="width: 80%;">
  </div>

</div>

Note: Explain fig well

----
<div class="image-block">
  <img src="images/flow_chart.svg" style="width: 48%;">
</div>

----
### Tested Architectures
<div class="tabular-container">

  <div class="tabular-row">
    <div class="model-name">EfficientNet-B0</div>
    <div class="model-param">4M</div>
    <div class="model-desc">Scaled CNN baseline</div>
  </div>

  <div class="tabular-row">
    <div class="model-name">DenseNet-169</div>
    <div class="model-param">12M</div>
    <div class="model-desc">Dense CNN, feature reuse</div>
  </div>

  <div class="tabular-row">
    <div class="model-name">ResNet-50</div>
    <div class="model-param">23M</div>
    <div class="model-desc">Residual blocks, deep</div>
  </div>

  <div class="tabular-row">
    <div class="model-name">ViT-B/16</div>
    <div class="model-param">85M</div>
    <div class="model-desc">Transformer, patch-wise</div>
  </div>

</div>

pretrained and from scratch

Note:
**EfficientNet-B0**<br>
Scales depth, width, and resolution in a balanced way<br>
Lightweight baseline with ~4 million parameters<br>
**DenseNet-169**<br>
Uses dense connections between layers (each layer receives input from all previous layers)<br>
Enables feature reuse and efficient gradient flow, ~12 million parameters<br>
**ResNet-50**<br>
Deep architecture with residual (skip) connections<br>
Robust performance, ~23 million parameters<br>
**ViT-B/16**<br>
Vision Transformer using 16×16 image patches<br>
Uses self-attention (each patch attends to all others, capturing global context), ~85 million parameters

----
### Cross Validation

<img src="images/k-fold.svg" style="width: 60%; margin-bottom: -50px;">

----
### Evaluation

<img src="images/eval.svg" style="width: 100%; margin-bottom: -50px;">

Note:  
Why different aggregation approaches — to retain information about distribution of fold performance.

**Balanced Accuracy**: Average of recall scores across classes. Useful for imbalanced datasets.  
**Accuracy**: Overall proportion of correctly classified samples. Can be misleading if classes are imbalanced.  
**Recall**: True positives / (true positives + false negatives). Measures how well a class is detected (sensitivity).  
**F1 Score**: Harmonic mean of precision and recall. Balances false positives and false negatives.  
**Support**: Number of ground truth samples per class. Indicates class distribution in the test set.

---
## Results

----
### Comparing Different Model Architectures

<div class="image-row">

  <div class="image-col">
    <div class="table-caption">Table: Balanced accuracy of all models – mean ± standard deviation; best values highlighted</div>
    <img src="images/table_compare.png">
  </div>

  <div class="image-col">
    <img src="images/bal_acc_img.jpg">
    <div class="figure-caption">Figure: Balanced accuracy across folds</div>
  </div>

</div>

----
### Pretrained EfficientNet-B0

<div class="image-row">

  <div class="image-col">
    <div class="table-caption">Table: Class-wise precision, recall, F1-score, and support for the pretrained EfficientNet-B0</div>
    <img src="images/table_best.png">
  </div>

  <div class="image-col">
    <img src="images/conf_matrix_best.jpg" style="width: 80%;">
    <div class="figure-caption">Figure: Confusion matrix EfficientNet-B0</div>
  </div>

</div>

----
### Stoats: hard to detect – easy to classify

<div class="image-block">
  <img src="images/mustela_hard_detect.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Not detected stoats</div>
</div>

----
### Stoats: hard to detect – easy to classify

<div class="image-block">
  <img src="images/mustela_easy_classify.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Classification examples</div>
</div>

----
### Looking into some errors

<div class="image-block">
  <img src="images/false_class_snails.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Detected snails – classified as mammals</div>
</div>

----
### Looking into some errors

<div class="image-block">
  <img src="images/no_detect_error.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Missed detections</div>
</div>

----
### Correlation?

<div style="margin-top: 0em; margin-bottom: -1em;font-size: 0.8em;">
  <p>
    Spearman’s rank correlation:
  </p>
  <ul style="list-style-type: none; padding-left: 0;">
    <li><em>Correctly classified samples</em>: <strong>ρ = 0.092</strong></li>
    <li><em>Incorrectly classified samples</em>: <strong>ρ = 0.276</strong></li>
  </ul>
</div>

<div class="image-block">
  <img src="images/pred_conf_hexbin.jpg" style="max-width: 600px;">
  <div class="figure-caption">Figure: Correlation of detection and classification confidence</div>
</div>

---
## Outlook

----
### Directions for Improvements

  <div class="image-col">
    <img src="images/improvements.svg" style="width: 1000%;">
  </div>

Notes:
- Introduce a non-target class for OOD detection  
- Add additional species, e.g. _Glis glis_  
- Improve detection quality, e.g. via fine-tuning  
- Explore sequence-aware or temporally informed classification approaches

----
### Automated Sequence Detection

<br><br>
Utilizing OCR for sequence detection:

<div class="image-block">
  <img src="images/ocr_example.jpg" style="max-width: 1000px;">
  <div class="figure-caption">Figure: Top strip of a random sample</div>
</div>

Output string was:

_2019-09-04 1:02:09 AM M 1/3 #9 10°C_

---
## Discussion

----
<iframe 
  width="90%" 
  height="600" 
  src="https://miro.com/app/live-embed/uXjVIhkMbMs=/?focusWidget=3458764633579243550&embedMode=view_only_without_ui&embedId=494225355739" 
  frameborder="0" 
  scrolling="no" 
  allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen>
</iframe>

---
## Feedback

Note:

- Fedback to my supervisors
- thank the audience
- feedback to me