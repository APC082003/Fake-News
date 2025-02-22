# Fake-News
Notre Dame University
Bangladesh
Artificial Intelligence Lab
CSE4104
Fake News Prediction
Submitted To :
Humayara Binte Rashid
Lecturer
Department of CSE
Submitted By :
Abrity Paul Chowdhury
ID : 0692130005101005
Batch : CSE 17
Submission Date: November 20, 2024Contents
0.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
0.2 Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
0.3 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
0.4 Related works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
0.4.1 Related works-1 . . . . . . . . . . . . . . . . . . . . . . . 2
0.4.2 Related works-2 . . . . . . . . . . . . . . . . . . . . . . . 3
0.5 Why consider it AI? . . . . . . . . . . . . . . . . . . . . . . . . . 4
0.6 Workflow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
0.7 Tools . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
0.8 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
0.9 Code & Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
0.10 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
10.1 Introduction
Fake news refers to false or misleading information presented as news, often
created to influence public opinion or generate web traffic. The spread of fake
news has become a significant concern, impacting societal trust and political
stability. With the growing volume of online news, developing models that can
accurately identify and filter out fake news is crucial to ensure the reliability of
information consumed by the public.
0.2 Objective
To develop a machine learning model that can accurately classify news articles
as real or fake, several key steps are involved. First, the news data must be
pre-processed and cleaned to ensure it’s suitable for analysis. Next, meaningful
features are extracted from the text using natural language processing techniques. These features are then used to train and evaluate machine learning
models to achieve optimal performance. Finally, different models are compared
to select the most effective one for accurately identifying fake news.
0.3 Motivation
The rapid spread of misinformation can have a profound social impact, undermining public trust in media and even influencing political outcomes. Given
the overwhelming volume of information, manually verifying news articles is
impractical, highlighting the need for automated detection systems that offer
a scalable and efficient solution. Advances in machine learning and natural
language processing have made the development of robust fake news detection
systems increasingly feasible, offering new ways to combat misinformation effectively.
0.4 Related works
0.4.1 Related works-1
Fake news detection using discourse segment structure analysis [5]
Contribution :
Here the author introduced a novel discourse-level approach for detecting fake
news using deep learning, leveraging a Bidirectional GRU to analyze the hierarchical structure of content. This approach achieved an accuracy of 74.62% and
an F1 score of 0.76, demonstrating its effectiveness in distinguishing fake news
based on discourse analysis.
Tools: Python, Google Colab, Word2Vec, GRU models
2Figure1:Stages of Approach
Limitations:
The model faces limitations due to a lack of dataset diversity, which risks introducing bias into the results. Its performance could be enhanced by incorporating
more advanced models or expanding data coverage to better represent various
news sources. Additionally, the model may struggle to generalize effectively
across different news contexts, potentially impacting its reliability in diverse
real-world scenarios.
0.4.2 Related works-2
Fake News Detection Using Machine Learning Approaches[1]
Contribution:
They developed a fake news detection model leveraging algorithms such as XGBoost, Random Forest, and Naive Bayes. To enhance the feature extraction
process, we applied NLP techniques like tokenization and TF-IDF. The model
reached an accuracy of up to 75% with its performance assessed using confusion
matrices.
Tools: Python, scikit-learn, NLTK, Anaconda, and the LIAR Dataset.
3Figure2:Working Procedure
Limitations :
The model’s accuracy is currently limited to 75% indicating room for improvement. Additionally, it may not generalize well beyond political news, limiting its
broader applicability. The model’s performance also relies heavily on effective
feature extraction, which is crucial for accurate classification.
0.5 Why consider it AI?
This project is considered an AI endeavor because it uses machine learning
techniques to automate the identification of fake and real news. It employs natural language processing (NLP) to process and analyze textual data, extracting
patterns and features that are not immediately obvious to humans. The system mimics human cognitive abilities by learning from vast amounts of labeled
news articles and generalizing this knowledge to predict the credibility of unseen
news. By training models like Naive Bayes, Logistic Regression, or advanced
neural networks, the AI can classify news articles with high accuracy. AI brings
scalability to this task, enabling the evaluation of thousands of articles in seconds, which would be impossible for humans to achieve manually. The project
demonstrates how AI can aid in combating misinformation, a critical challenge
in today’s digital age. It not only reduces human workload but also improves
the efficiency and reliability of misinformation detection systems. The use of
4advanced AI techniques, such as deep learning and transformers, further enhances the model’s ability to understand complex patterns in language. This
real-world application of AI has significant societal impact, helping maintain
trust in journalism and curbing the spread of false information. By automating decision-making in this domain, the project exemplifies the power of AI in
addressing pressing global challenges.
0.6 Workflow
Here is the procedure that will be followed by the proposed approach .
Figure3:Workflow of proposed project
0.7 Tools
Python Libraries [4]
The project utilizes several Python libraries to handle various tasks. Pandasis
used for data manipulation and pre-processing, while NumPy supports numerical operations and array handling. Scikit-learn is employed for model
building, data splitting, and performance evaluation. NLTK facilitates natural
language processing tasks, including tokenization and stopword removal. Additionally, the re library is used for regular expressions to clean and process text
efficiently.
Machine Learning Models [3]
The model uses several algorithms for classification tasks.Logistic Regression
serves as a common baseline for binary classification problems. The Naive
Bayes Classifier is often employed for text classification tasks due to its simplicity and efficiency. Support Vector Machines (SVM) are utilized for
better performance on certain datasets, especially when the data is not linearly
separable. Random Forest is used for ensemble-based predictions, providing
greater robustness and accuracy by combining multiple decision trees.
Text Processing Tools [2]
The model utilizes several techniques for text pre-processing and feature extraction. CountVectorizer is used to convert text into a matrix of token
counts, representing the frequency of each word in the document. TfidfVectorizer transforms the text into Term Frequency-Inverse Document Frequency
5(TF-IDF) features, which help weigh words based on their importance across
different documents. Additionally, NLTK is employed for text processing and
cleaning tasks, such as removing stop words and performing stemming to reduce
words to their base form.
0.8 Methodology
The proposed diagram
Figure4:Diagram of proposed process of fake news prediction
The updated diagram
Figure5:Diagram of updated process of fake news prediction
60.9 Code & Result
Dataset [6]
Figure6:Code for Dataset
Figure7:Dataset in Kaggle
7Main Code
Figure8:Installing kaggle and downloading dataset in google
colab part1
Figure9:Installing kaggle and downloading dataset in google
colab part2
8Figure10:Importing dependencies
Figure11:Data pre processing
9Figure12:Checking missing values and column merging
Figure13:Separating data and label
10Figure14:Stemming
Figure15:Stemming function used in content
11Figure16:Separating content and label
Figure17:Tfidfvectorizer
12Figure18:Splitting and training
Figure19:Accuracy
13Figure20:Prediction
0.10 Conclusion
The project aims to leverage machine learning and NLP techniques to build an
efficient fake news detection system. The model can be improved with more
diverse data and advanced algorithms like deep learning .As there are some
limitations on logistic regression as well as working with real time data . There
are some work that needs to be done. So, in future i will work to improve in
this specific sectors . Hopefully it will enhance the reliability of information and
contributes to combating the spread of misinformation.
14Bibliography
[1] Zeba Khanam, BN Alwasel, H Sirafi, and Mamoon Rashid. Fake news detection using machine learning approaches. In IOP conference series: materials
science and engineering, volume 1099, page 012040. IOP Publishing, 2021.
[2] John Levine. Flex & Bison: Text Processing Tools. ” O’Reilly Media, Inc.”,
2009.
[3] Congzheng Song, Thomas Ristenpart, and Vitaly Shmatikov. Machine learning models that remember too much. In Proceedings of the 2017 ACM
SIGSAC Conference on Computer and Communications Security, CCS ’17,
page 587–601, New York, NY, USA, 2017. Association for Computing Machinery.
[4] I. Stanˇcin and A. Jovi´c. An overview and comparison of free python libraries
for data mining and big data analysis. In 2019 42nd International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), pages 977–982, 2019.
[5] Anmol Uppal, Vipul Sachdeva, and Seema Sharma. Fake news detection
using discourse segment structure analysis. In 2020 10th International Conference on Cloud Computing, Data Science & Engineering (Confluence),
pages 751–756. IEEE, 2020.
[6] Real time news. (2024, November 18). Kaggle.
15
