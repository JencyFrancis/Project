Predictive Maintenance with Oversampling Techniques
Research Objective
Does oversampling imbalanced data improve the performance of Random Forest, MLP Classifier, and KNN in predicting machine failure from sensor data?
Research Question
To evaluate the impact of oversampling techniques (SMOTE, ADASYN, RandomOverSampler, SMOTETomek) on the performance of Random Forest, MLP Classifier, and KNN in predicting machine failures from imbalanced sensor data, measured by F1-score, recall, and ROC AUC.
Project Summary
This MSc Data Science research project provides a comprehensive evaluation of oversampling techniques for addressing class imbalance in predictive maintenance applications. Using the AI4I 2020 Predictive Maintenance Dataset, the study systematically compares the effectiveness of four oversampling methods across three machine learning algorithms.
Key Findings
•	Random Forest + RandomOverSampler achieved best overall performance (F1-score: 0.7667)
•	Substantial recall improvements across all algorithms (up to 120% for KNN)
•	Oversampling techniques consistently improve minority class detection capability
•	RandomOverSampler demonstrates surprising effectiveness compared to sophisticated synthetic methods
Quick Start
Prerequisites
•	Python 3.8+
•	Conda or pip for package management 
       
Dataset Information
Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
•	Source: UCI Machine Learning Repository
•	Size: 10,000 observations
•	Features: 6 (Type, air temperature, process temperature, rotational speed, torque, tool wear)
•	Target: Binary machine failure classification
•	Class Imbalance after preprocessing: 29.2:1 ratio (96.7% normal, 3.3% failures)
Methodology
Machine Learning Algorithms
1.	Random Forest - Ensemble method with natural robustness to imbalanced data
2.	Multi-Layer Perceptron (MLP) - Neural network for complex pattern recognition
3.	K-Nearest Neighbors (KNN) - Distance-based algorithm for local similarity patterns
Oversampling Techniques
1.	SMOTE - Synthetic Minority Oversampling Technique
2.	ADASYN - Adaptive Synthetic Sampling
3.	RandomOverSampler - Simple random duplication
4.	SMOTETomek - Hybrid oversampling with undersampling
Evaluation Metrics
•	F1-Score - Harmonic mean of precision and recall
•	Recall - Proportion of actual failures correctly identified
•	ROC AUC - Area under the receiver operating characteristic curve
Notebooks Workflow
1.	Data Preprocessing - Data cleaning, structural checks, type conversion, validation and statistical summary
2.	Data Exploration - Dataset analysis and visualization
3.	Baseline Models - Initial model training without oversampling
4.	Oversampling Analysis - Systematic oversampling technique evaluation
5.	Hyperparameter Tuning - Optimization of best combinations
6.	Learning Curves - Model behaviour and overfitting analysis
7.	Final Evaluation - Test set performance and statistical validation
8.	Model Analysis - Feature importance and insights

Dependencies
Core Libraries
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
Additional Tools
jupyter>=1.0.0
scipy>=1.7.0
plotly>=5.0.0
statsmodels>=0.12.0

Contributing
Contributions are welcome!
License
This project is licensed under the MIT License.
Citation
If you use this work in your research, please cite our study on Does oversampling imbalanced data improve the performance of Random Forest, MLP Classifier, and KNN in predicting machine failure from sensor data?
.
Contact
•	Author: Jency Francis
•	Email: francis.jency01@gmail.com
Acknowledgments
•	UCI Machine Learning Repository for providing the AI4I 2020 Predictive Maintenance Dataset
•	imbalanced-learn development team for oversampling implementations
•	scikit-learn community for machine learning tools
•	Academic supervisors and reviewers for guidance and feedback










<img width="451" height="687" alt="image" src="https://github.com/user-attachments/assets/a7b972dd-997e-443f-a99b-0889b8e22124" />
