# NS-CausalKT: Project Explanation

This document provides a comprehensive deep dive into the **NS-CausalKT (Neural-Symbolic Causal Knowledge Tracing)** system.

---

## 1. What is this project?
At its core, this project is an **Intelligent Tutoring System (ITS)**. It aims to solve the problem of "Knowledge Tracing"—tracking a student's mastery of different concepts over time based on their history of answering questions.

Unlike traditional AI models that just say "the student will fail," our model explains **WHY** they failed using Causal Inference.

---

## 2. Dataset Deep Dive
### Kind of Dataset
We use the **ASSISTments 2009** dataset. This is a longitudinal dataset of student interactions with an online math tutoring platform. It is a "time-series" dataset where each row represents a student answering a specific math problem at a specific time.

### Key Features
- **user_id**: Unique ID for each student.
- **skill_name**: The math concept being tested (e.g., "Linear Equations", "Fractions").
- **correct**: 1 if the student got it right, 0 if they got it wrong.
- **ms_first_response**: Time spent (in milliseconds) before the first answer.
- **hint_count**: Number of hints the student asked for.

### Purpose
The purpose of this dataset is to provide the "ground truth" for how students learn. By looking at thousands of past students, the model learns that "Fractions" is usually a prerequisite for "Algebra."

### Data Sample
| order_id | user_id | skill_name | correct | ms_first_response |
|----------|---------|------------|---------|-------------------|
| 33022537 | 64525   | Box and Whisker | 1 | 32454 |
| 33022709 | 64525   | Box and Whisker | 1 | 4922 |
| 35450204 | 70363   | Box and Whisker | 0 | 25390 |

---

## 3. Comparison: DKT vs. AKT vs. NS-CausalKT

| Feature | **DKT (Baseline)** | **AKT (Baseline)** | **NS-CausalKT (Ours)** |
|---------|-------------------|-------------------|------------------------|
| **Core Tech** | LSTM (Recurrent) | Transformer (Attention) | Neuro-Symbolic Transformer |
| **Interpretation**| None (Black Box) | Attention Weights | Causal ACE Rankings |
| **Logic** | Purely Statistical| Purely Statistical | Symbolic Logic Rules |
| **Interventions** | No | No | **Yes** (do-calculus) |
| **Accuracy (AUC)**| ~0.78 - 0.80 | ~0.82 - 0.84 | **~0.81 - 0.85** |

### How they work:
- **DKT (Deep Knowledge Tracing)**: Uses an LSTM to remember everything a student did. It's like a memory bank that updates with every answer.
- **AKT (Attentive Knowledge Tracing)**: Uses "Attention" to look back at specific past questions that are similar to the current one. It ignores irrelevant history.
- **NS-CausalKT**: It uses the speed of a Transformer (like AKT) but adds a **Causal Layer**. It doesn't just look for patterns; it looks for **Cause-and-Effect**.

---

## 4. How NS-CausalKT Works (The Three Layers)

### 1. Neural Layer (The Brain)
It uses a **Transformer Encoder** to process the sequence of student answers. It converts the history into a "Latent Knowledge State"—a 128-dimensional vector representing the student's mind.

### 2. Symbolic Layer (The Rules)
It uses a **Concept Dependency Graph (DAG)**. This is a "map" of math. It tells the model: "If you don't know Linear Equations, you CANNOT know Quadratic Functions." This enforces logical consistency.

### 3. Causal Layer (The Scientist)
It uses **Pearl's do-calculus**. It simulates interventions. It asks: "If I teach this student Fractions right now, how much will their Algebra score improve tomorrow?" This is called **Average Causal Effect (ACE)**.

---

## 5. How we trained the model
1. **Preprocessing**: We took 400,000 interactions and grouped them by student.
2. **Loss Functions**: We didn't just train on "correctness." We used three losses:
   - **Prediction Loss**: Making sure the model predicts the right answer.
   - **Symbolic Loss**: Penalizing the model if it breaks the rules of the math map.
   - **Causal Loss**: Making sure the simulated interventions match real-world correlations.
3. **Hardware**: Trained on **CUDA-enabled GPUs** using the AdamW optimizer for 30 epochs.

---

## 6. What is it predicting at the end of the day?
At any given moment, the model outputs:
1. **The Probability (0 to 1)** that the student will get the *next* specific question right.
2. **The Mastery Level (0% to 100%)** for every single math concept in the system.
3. **The Root Cause**: If the student is failing, it identifies which foundational concept is missing.

---

## 7. Sample Walkthrough
1. **Step 1: Dashboard**: You open the dashboard and see the model's accuracy is 76%. You select "Student 14" and see their mastery of "Fractions" is dropping.
2. **Step 2: Causal Analysis**: You move the "Fractions" slider in the Counterfactual Sandbox. You see that increasing Fractions by 20% would boost their overall score by 8%.
3. **Step 3: Agentic Grading**: You upload an image of the student's handwritten homework.
4. **Step 4: AI Analysis**: GPT-4o Mini (the agent) looks at the image, identifies where the student made a sign error, and combines this with the NS-CausalKT data to say: "The student made an error here because their foundational understanding of Linear Equations is weak."

---
