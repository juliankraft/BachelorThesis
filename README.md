## Deep Learning for Biodiversity Monitoring: Automated Classification of Small Mammals Captured in Foto Trap Boxes

### Keywords
Camera Trapping, Deep Learning, Transfer Learning, Object Detection, Image Classification, Small Mammals, Biodiversity Monitoring, MegaDetector

### Abstract

This thesis explores the use of Deep Learning (DL) to automate the classification of small mammals captured in camera trap images gathered as part of the Wildlife@Campus project.
A dataset of over 400,000 labeled images, grouped into sequences, was processed using MegaDetector to filter and crop relevant regions of interest.
Several model architectures were evaluated, with the pretrained EfficientNet-B0 achieving the highest balanced accuracy of 0.992 for the classification task.
A comprehensive data pipeline was developed, including detection, preprocessing, cross-validation and classification on an image and sequence level, enabling efficient and reproducible model training and evaluation.
While pretrained models outperformed non-pretrained variants, the results also demonstrated that smaller architectures can be accurate while saving resources.
The study highlights the importance of detection quality, label accuracy and the need for a non-target class to handle unknown species such as snails or misdetections such as plant parts or simply empty images.
In addition to the missing non-target class, the study also emphasizes the need for an improved detection process in order to reduce missed sightings.
There is still a high dependency on large amounts of labeled data, which is a real challenge when adding additional classes.
This work lays the foundation for integrating DL into the camera trap approach of the Wildlife@Campus project, aiming to reduce the associated manual effort in small mammal monitoring and contribute to ecological research.

### Project Details

**Author:** Julian Kraft  
**Supervisors:** Dr. Stefan Glüge, Dr. Matthias Nyfeler  
**Institution:** Zurich University of Applied Sciences (ZHAW)  
**Program:** BSc Natural Resource Sciences  
**Project:** Bachelor’s Thesis  
**Date:** 2025-06-24  

|                   |                                            |
|-------------------|--------------------------------------------|
| **Author**        | Julian Kraft                               |
| **Supervisors**<br><br>   | Dr. Stefan Glüge<br>Dr. Matthias Nyfeler   |
| **Institution**   | Zurich University of Applied Sciences (ZHAW) |
| **Program**       | BSc Natural Resource Sciences              |
| **Project**       | Bachelor’s Thesis                          |
| **Date**          | 2025-06-24                                 |
|                   |                                            |

## Repository Content

This repository contains all relevant code, the LaTeX source and all visualizations created during the thesis.

**Thesis:** [main.pdf](./LaTeX/main.pdf)<br>
**Visualizations:** [visualizations.ipynb](./visualizations.ipynb)

### Repository Structure

- `LaTeX/`: LaTeX source code of the thesis  
- `code/`: All Python code developed for this project  
- `run/`: Files used to run the experiments

### Environment

The environment used for running the models and evaluations was created using Anaconda.

- **With CUDA support:** [environment_cuda.yml](./environment_cuda.yml)  
- **For macOS:** [environment_osx.yml](./environment_osx.yml)

To install the code as a developer package, run:

```bash
pip install -e ./code
```

## Acknowledgements

This project makes use of [MegaDetector](https://github.com/agentmorris/MegaDetector/tree/main), an open-source object detection model developed by Microsoft AI for Earth. 
It was used to automatically detect and crop regions of interest in camera trap images as a preprocessing step.
MegaDetector is licensed under the [MIT License](https://github.com/agentmorris/MegaDetector/blob/main/LICENSE). 

## License

This repository is licensed under the **CC0 1.0 Universal (Public Domain Dedication)**. 

To the extent possible under law, the authors of this repository have waived all copyright and related or neighboring rights to this work. 

For more details, see the [LICENSE](./LICENSE) file or visit the [Creative Commons Legal Code](https://creativecommons.org/publicdomain/zero/1.0/legalcode).


