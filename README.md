# Information-Extraction-Pretrained-Models
![](https://github.com/seanaba/Information-Extraction-Pretrained-Models/blob/master/doc/pic/pic1.jpg)

Information extraction from unstructured documents is a real challenge for scientists. Fortunately, there are several pre-trained name entity recognition models to derive general information from a text. Location, date, money, and time are examples of the general information which can be derived from the pre-trained libraries. NLTK, Spacy, and StanfordCoreNLP are three common pre-trained python libraries for name entity recognition. The StanfordCoreNLP is actually in Java but it can be run by Python. 
The core algorithm to train the name entity recognition models for all the mentioned libraries is Conditional Random Fields (CRF) which is a sequential models, but other sequential algorithms such as Recurrent Neural Networks (RNN) can be used to train the models.
In the following, an example is presented to show the effectiveness of each model to extract general information. To be specific, the following text is provided to test the pre-trained models and geographic location, date, and money are the information targeted to be extracted from these three pre-trained models.
```text
This document is for testing information extraction methodologies. The statue of liberty is located in New York and dedicated in 10-28-1886. The governor of New York voted a bill to provide $50000 for the statue project in 1884. Let us find out which pre-trained information extraction methodology is able to extract date and money precisely.
```
The results from the models to extract date, money, and location are provided in the following.
```python
NLTK
['New York', 'GPE']
```
```python
Spacy
['New York', 'GPE']
['10-28-1886', 'DATE']
['New York', 'GPE']
['50000', 'MONEY']
['1884', 'DATE']
```
```python
StanfordCoreNLP
['New York', 'Location']
['10-28-1886', 'DATE']
['New York', 'Location']
['50000', 'MONEY']
['1884', 'DATE']
```
The results demonstrate the effectiveness of both Spacy and StanfordCoreNLP for our simple example. Clearly, the effectiveness of the algorithm depends on type of text that is targeted to be used as well as the information which is required to be extracted.
