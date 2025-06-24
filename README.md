## Deep Learning for Biodiversity Monitoring: Automated Classification of Small Mammals Captured in Foto Trap Boxes

### Keywords
    Camera Trapping, Deep Learning, Transfer Learning, Object Detection, Image Classification, Small Mammals, Biodiversity Monitoring,MegaDetector

### Abstract

Understanding the distribution of impervious and pervious surfaces is critical 
for effective urban planning, environmental management, and rainfall impact analysis. 
This study explores the use of convolutional neural networks (CNNs) for 
pixel-based classification of aerial remote sensing data to assess surface sealing. 
Using high-resolution SwissImage RS data, the analysis employs a simplified 
ResNet-18 architecture adapted for four-channel inputs, including RGB and 
near-infrared bands. A detailed workflow was developed, describing the 
preprocessing of the data, the training, and the evaluation of the model. 
A data augmentation strategy was implemented to improve the model's performance, 
and a hyperparameter tuning process was conducted to optimize the model. 
The best-performing model achieved a classification accuracy of 0.927, 
which is in a similar range to the results of previous studies 
utilizing a geoprocessing approach. While challenges such as mixed-pixel 
problems or limited data availability remain, this study demonstrates 
the potential of deep learning for detailed surface sealing analysis.

**Author:**         Julian Kraft<br>
**Supervisor:**     Dr. Stefan Glüge<br>
                    Dr. Matthias Nyfeler<br>
**Institution:**    Zurich University of Applied Sciences (ZHAW)<br>
**Program:**        BSc Natural Resource Sciences<br>
**Project:**        Bachelors Thesis<br>
**Date:**           2025-06-24

**Paper:** [link](./LaTeX/main.pdf)<br>
**Visualizations:** [link](./code/eval/visualizations.ipynb)

### Repository Content

This repository provides all the relevant code as well as the LaTeX source code and
and all visualizations used in the thesis.

### Repository Structure

- `LaTeX/`: LaTeX source code of the term paper
- `code/`: all Python code created for this thesis
- `experiments/`: 

### Environment

The environment used to run this model and the evaluation was created using Anaconda.
**With Cuda-support:**  `environment_cuda.yml`
**For OSx:**            `environment_osx.yml`


Additionally, the code developed in this project can be installed as a developer package. To do so, run:

```bash
pip install -e ./code/
```

### License

This repository is licensed under the **CC0 1.0 Universal (Public Domain Dedication)**. 

To the extent possible under law, the authors of this repository have waived all copyright and related or neighboring rights to this work. 

For more details, see the [LICENSE](./LICENSE) file or visit the [Creative Commons Legal Code](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
