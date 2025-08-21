# Machine Learning Samples, My Personal Training 🚀

This repository is a personal collection of machine learning practice projects. Each folder contains a simple implementation of a specific model, along with the dataset used. The goal is to learn, experiment, and build a reference for different ML algorithms using Python, scikit-learn, pandas, and numpy.

The repository is structured so that each model has its own folder with the script and dataset. For example:
ML_sample/
├── NaiveBayes/
│ ├── naive_bayes.py
│ └── dataset.csv
├── DecisionTree/
│ ├── decision_tree.py
│ └── dataset.csv
├── RandomForest/
│ ├── random_forest.py
│ └── dataset.csv
├── KNN/
│ ├── knn.py
│ └── dataset.csv
├── LogisticRegression/
│ ├── logistic_regression.py
│ └── dataset.csv
├── NeuralNetwork/
│ ├── neural_network.py
│ └── dataset.csv
└── README.md


Models currently included are Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors, Logistic Regression, and a basic Neural Network. The idea is to gradually extend this with more models like SVM, Gradient Boosting, and XGBoost, and to also compare models on the same dataset with added visualizations and notebooks.

To run any model, install the requirements first:

```bash
pip install -r requirements.txt

where requirements.txt contains:
numpy
pandas
scikit-learn

Then navigate into the folder of the model you want to run and execute the script, for example:
cd NaiveBayes
python naive_bayes.py

This repository is mainly for personal learning and tutorials, a way to keep track of experiments and serve as a reference.
