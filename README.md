# Cyberbullying Detection and Classification

## Project Overview
This project aims to develop an advanced application for detecting and classifying cyberbullying on social media platforms using Natural Language Processing (NLP) techniques and machine learning. The goal is to create a safer online environment by identifying and addressing harmful content effectively.

## Objectives
- Detect instances of cyberbullying in social media posts and comments.
- Classify detected instances into specific types such as "ethnicity," "gender," "religion," etc.
- Achieve a high prediction accuracy for reliable and efficient cyberbullying detection.

## Key Features
- **Advanced Machine Learning Model:** Utilizes a Linear Support Vector Classification (LinearSVC) model trained on a well-balanced dataset.
- **Comprehensive Data Preprocessing:** Includes data cleaning, tokenization, and lemmatization to prepare text data for analysis.
- **User-Friendly Interface:** Developed using Jupyter Notebooks for easy interaction and testing.

## Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, spaCy, Matplotlib, NumPy, Scikit-learn
- **Environment:** Anaconda, Jupyter Notebooks

## Dataset
The dataset used for this project is sourced from Kaggle, titled "Cyberbullying Classification" by J. Wang, K. Fu, and C.T. Lu. It includes tweet texts and corresponding cyberbullying types, ensuring comprehensive representation.

## Methodology
1. **Data Acquisition:** Download and load the dataset.
2. **Data Preprocessing:** Clean the text data by removing HTML tags, URLs, mentions, hashtags, emojis, non-Unicode characters, numbers, punctuation, non-English characters, and extra spaces. Tokenize and lemmatize the text using spaCy.
3. **Data Partitioning:** Split the data into training and testing sets.
4. **Model Training:** Train a LinearSVC model on the preprocessed text data.
5. **Model Testing:** Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Installation and Setup
### Step 1: Download and Install Miniconda
1. Visit the Miniconda website: https://docs.conda.io/en/latest/miniconda.html
2. Download the appropriate Miniconda installer for your operating system.
3. Run the installer and follow the on-screen instructions to install Miniconda.

### Step 2: Set Up the Project Environment
1. Download and extract the project files from the repository.
2. Open the Anaconda Command Prompt and navigate to the project directory using the `cd` command. For example:
   ```bash
   cd C:\path\to\project
   ```

### Step 3: Create and Activate the Conda Environment
1. Create a new conda environment with the required packages:
   ```bash
   conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter spacy
   ```
2. Activate the environment:
   ```bash
   conda activate ./env
   ```
3. Install the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Step 4: Launch Jupyter Notebook
1. In the Anaconda Command Prompt, launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the `Cyberbullying_Detection_Classification.ipynb` notebook and run all cells to execute the project code.

## Results
- **Accuracy:** The model achieved an accuracy rate of 90.31%.
- **Performance Metrics:** High precision, recall, and F1-score values for each cyberbullying category.
- **Confusion Matrix:** Visual representation of the model's performance.

## Screenshots
### Confusion Matrix

### Data Preprocessing Steps

### Model Performance Metrics

## Usage
Users can input text from posts or comments into the application to detect potential cyberbullying. The model will analyze the text and predict whether it contains cyberbullying and its specific type.

## Future Work
- Explore additional NLP techniques and models to further improve detection accuracy.
- Integrate the application into real-time social media monitoring systems.
- Extend the classification to include more types of cyberbullying and related harmful behaviors.
