# Advanced_IA_implementation

## "Analítica de datos y herramientas de inteligencia artificial" & "Inteligencia Artificial para la Ciencia de Datos"

## Abstract

This repository stores all the notes and exercises I completed in my "Artificial Intelligence for Data Science" course. The main topics covered were machine learning with supervised and unsupervised training, and the implementation of deep learning in CV, NLP, RL, and brain signal analysis.

### Pedro Mauri Martínez - A01029143

---

## Repository Structure

```text
Advanced_IA_implementation/
│
├── Deep Learning/
│   │
│   ├── Brain Signals/
│   |   ├── Kalman_Filter_Manual.ipynb
│   |   └── Sine_Wave.ipynb
│   │
│   ├── CV & PyTorch/
│   |   ├── CV - Computer Vision/
│   |   │   ├── ...
│   |   │   └── ...
│   |   │
│   |   ├── PyTorch/
│   |   │   ├── ...
│   |   │   └── ...
│   |   │
│   |   └── CV_Final_Exam.ipynb
│   │
│   ├── Datasets/
│   |   ├── orders-1.csv
│   |   ├── originalTest.txt
│   |   ├── originalTraining.txt
│   |   ├── the_count_of_montecristo.txt
│   |   └── User1_Pre2.csv
│   │
│   ├── MLOps/
│   |   ├── (TextBook) Designing Machine Learning Systems.pdf
│   |   ├── MLOps Final Assesment.pdf
│   |   ├── Feature_Engineering.ipynb
│   |   └── MLOps Principles - Homework.pdf
│   │
│   ├── NLP - Natural Language Processing/
│   |   ├── NLP_Final_Assessment.ipynb
│   |   └── NLP_Final_Exam.ipynb
│   |
│   ├── RoboCap/ (GitHub Submodule)
│   │
│   └── RL - Reinforcement Learning/
│       ├── Act1__TabularQ.py
│       ├── Act2__PolicyGradients (incomplete).py
│       ├── Class1__Expected_Value.ipynb
│       ├── Class2__Maximum_Likelihood.ipynb
│       ├── Class3__Categorical_mle.ipynb
│       └── Class4__Policy_Gradients__Reducing_Variance.ipynb
│
├── Machine Learning/
│   │
│   ├── .../
│   |   ├── ...
│   |   └── ...
│   │
│   ├── WinLosePredictor/ (GitHub Submodule)
│   │
│   ├── .../
│   |   ├── ...
│   |   └── ...
│   │
│   ├── .../
│   |   ├── ...
│   |   └── ...
│   │
│   └── Class Notes.pdf
│
└──requirements.txt (dependencies)
│
└── .gitignore
└── README.md
```

---

## Directories Overview

### Deep Learning

[description]

---

### Machine Learning

[description]

---

## Setup

1. Clone Repository

    ```bash
    git clone https://github.com/Pedro3107dev/Advanced_IA_Implementation.git
    cd Advanced_IA_Implementation
    ```

2. Create and activate Python's environment (PowerShell)

    ```bash
    python -m venv env
    .\env\Scripts\Activate.ps1
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

- Remember that whenever you want to run a .ipynb file, you must select the correct kernel in the top-right corner of the notebook or use the cell selector to use the correct environment.

---

## Profesors

**Machine Learning:**

- Víctor Manuel de la Cueva Hernández.
  - Manual implementation of algorithms.
- Dr. Esteban Castillo Juárez.
  - Implementation and use of scikit-learn: machine learning in Python module.
- Dr. Gualberto Aguilar Torres.
  - Neural Networks: Perceptron and associative networks.

**Deep Learning:**

- Dr. Esteban Castillo Juárez.
  - **NLP (Natural Language Processing):** Field of artificial intelligence that enables computers to understand, interpret, manipulate, and generate human language, whether written or spoken.
- Edoardo Bucheli Susarrey.
  - **MLOps (Machine Learning Operations):** Set of practices that automates and simplifies the development, deployment, and maintenance of machine learning models.
- Lizbeth Peralta Malváez.
  - **Brain Signals Processing:** Transforms raw, noisy electrical or magnetic activity from the brain into clean, usable data for medical diagnosis, research
- Octavio Navarro Hinojosa.
  - **PyTorch Usage for CV (Computer Vision):** Field of artificial intelligence that enables computers to extract information, identify objects, and understand images or videos in the same way as human vision.
- Gerardo Jesús Camacho González.
  - **RL (Reinforcement learning):** Method where an artificial intelligence program learns to make decisions by trying things and getting rewards or penalties
