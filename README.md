# school-notification-classification-task

## #01 Introduction 

### 💡 The neccessity of the project

Users often find it difficult to quickly and accurately obtain the information they need due to ambiguous category boundaries in school announcements, such as academic affairs, career, and scholarships. In particular, the main user group of the school website—students—are primarily interested in announcements about activities that can enhance their career prospects and qualifications.

➡️ To address this issue, we aim to build a classification model that categorizes announcements into two groups: those that contribute to building a resume (1) and those that do not (0).

## #02 Data Crawling

Defined a title scraping function by using `BeautifulSoup` to extract specific sections of the target webpage. Inspected the website's developer mode to identify the relevant tags and classes for the data, then extracted the necessary content using the find or find_all methods. We labeled 9,782 training samples and 976 test samples manually after scraping.

## #03 EDA

- Korean Font (Nanum Font) Installation
- Check for Missing Values & Remove Duplicates
- Check Class Balance: Since class imbalance in the training data may require data manipulation techniques such as SMOTE, oversampling, or undersampling, it is an important factor to examine.
- Analyze Title Length: Verified that text lengths do not significantly differ across classes.
- Visualize Dominant Words per Class Using a Word Cloud: Used a word cloud to visually identify which words are dominant in each class.

## #04 Preprocessing

- Removed 95% of crawled school names to prevents the model from overfitting to specific school names.
  
- Retained only English and Korean in titles, removing all numbers, parentheses, and special characters.

- Defined Custom Words and Stop Words to prevented incorrect tokenization of key terms. For example, the word **"장학생" (scholarship student)** was mistakenly split into **"장" (chapter) and "학생" (student)**. We created a **custom dictionary** and Komoran morphological parser to handle such cases properly.

## #05 Modeling

### Embedding

---

- N-gram Vectorization captures contextual information by considering the surrounding words.

- Utilize the last hidden state (embedding) obtained from KLUE-BERT-Base Tokenizer, which is a pre-trained model, to generate more meaningful embeddings.

➡️ Finally, both embeddings are stacked into the `X_combined` variable, which is fed to machine learning models. 

### Machine Learning Model

| Model | Random Forest | Multinomial Naive Bayes | Bernoulli Naive Bayes | Logistic Regression |
| --- | --- | --- | --- | --- |
| **Accuracy** | 0.9183 | 0.9008 | 0.9088 | 0.9321 |
| **F1 Score** | 0.9183 | 0.9008 | 0.9088 | 0.9321 |

### Deep Learning Model

### Multilingual BERT

- Multilingual Bert (m-bert) is a pre-trained bert-based model which was trained to understand multiple languages and generalized language representations. It was trained with self-supervised fashion without any ground-truth labels. 

### KLUE-BERT-BASE

- KLUE (Korean Language Understanding Evaluation) is a collection of benchmarks for 8 Korean NLU tasks and KLUE-BERT-BASE (klue-bert-base) is a pre-trained model with KLUE benchmarks.

  ## Experiment Results


| Model | Average Test Accuracy | Test F1 Score |
| --- | --- | --- |
| KLUE/BERT-base | 0.9217 | 0.9032 |
| Multilingual BERT | 0.9054 | 0.8838 |

