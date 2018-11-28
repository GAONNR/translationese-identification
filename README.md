# Graduation Research: Translationese Identification via Deep Learning

2018 Fall, 졸업 연구: 딥 러닝을 활용한 번역체(Translationese)의 Identification

## 1. 번역체(Translationese)란?

Translationese(번역투, 번역체)란, 'interlanguage', 'third code' 라고도 불리며, 어떤 언어 B로 쓰여진 문장이 다른 언어 A로 번역되었을 때, 언어 A에서 잘 나타나지 않는, 번역된 문장 특유의 특징을 가지는 것을 의미합니다. 오역(mistranslation)과 다른 점은, translationese에서는 오역에서 나타나는 단어의 잘못된 번역이나 문법적 오류가 나타나지 않아 언뜻 보기에는 완벽한 번역으로 보일 수 있으나, 원어민이 보기에는 잘 쓰이지 않는 표현이 사용되어 위화감이 느껴진다는 점이 있겠습니다.

## 2. 본 연구의 목표

Ella Rabinovich, Shuly Wintner의 2015년 논문 등에 따르면 SVM 등의 Supervised Learning을 활용한 Classifier의 경우 90%에 육박하는 상당히 높은 정확도를 보였으나, A language to B language로 작성된 모델을 C language에 적용하면 정확도가 떨어지고, 사용한 corpus와 다른 genre를 가지는 문장에 적용하여도 정확도가 현저히 떨어지는 문제가 있었습니다. 해당 논문에서는 이를 해결하기 위해 K-means 알고리즘을 위주로 한 Unsupervised Learning을 진행하였는데, 앞선 case에 대한 정확도가 전체적으로 향상되었으나, 그 정확도가 70% ~ 90% 사이에서 들쭉날쭉하여 큰 효과를 발휘하지 못하는 케이스 또한 존재하였습니다. 이에 Deep Learning을 활용한 Classifier를 모델링한다면 그 효과가 어떨지에 대한 궁금증을 갖게 되었는데, scholar.google.com 에서의 검색 결과로는 Deep Learning을 활용한 Classifier에 관한 논문을 찾지 못하였습니다. 따라서 본 연구에서는, Deep Learning을 활용한 Classifier를 모델링하고자 합니다.

## 3. Sources

- Functional words list from _On the features of translationese_, _Volansky et al_, 2015. Appendix A.4

- [Europarl English-French Parallel Corpus](http://cl.haifa.ac.il/projects/translationese/index.shtml)

## 4. Requirements

- Python 3.6
- virtualenv
- [Parallel Corpus](http://cl.haifa.ac.il/projects/translationese/index.shtml) (in `./corpus/`)

## 5. Usage

- Setup

  ```bash
  pip install spacy
  python -m spacy download en
  pip install -r requirements.txt
  ```

### Classical ML method (SVM)

- Get chunks & features of the corpus

  ```bash
  python chunknizer.py <corpusname>
  ```

- Get only features of the corpus

  ```bash
  python chunknizer.py <corpusname> features
  ```

- [supervised_classifier.py](./supervised_classifier.py) (Using SVM)

  ```bash
  python supervised_classifier.py <corpusname>
  python supervised_classifier.py <corpusname != europarl> cross # test between input corpus & europarl
  ```

### LSTM

- Get chunks of the corpus

  ```bash
  python words_to_numbers.py <corpusname or all>
  ```

- [lstm_classifier.py](./lstm_classifier.py)

  ```bash
  python lstm_classifier.py --corpus <corpusname=europarl> --max_words <max_words=500> --cross <cross corpus name>
  # all three parameters are not necessary. I recommend you to leave max_words parameter empty.
  ```

## 6. Results(TBU)

- Number of Tokens / Sentences

  | Data            | Sentences |    Tokens | Chunks(SVM) | Chunks(LSTM) |
  | --------------- | --------: | --------: | ----------: | -----------: |
  | Europarl - EN   |   217,421 | 5,979,208 |        2964 |         5208 |
  | Europarl - FR   |   130,051 | 4,037,457 |        1999 |         3531 |
  | Literature - EN |   217,421 | 5,979,208 |         446 |          794 |
  | Literature - FR |   130,051 | 4,037,457 |        1066 |         1785 |

- Accuracy

  | Corpus     | SVM | LSTM |
  | ---------- | --: | ---: |
  | Europarl   | 95% |  91% |
  | Literature | 96% |  92% |

- Cross Accuracy

  | Corpus 1 | Corpus 2   | SVM | LSTM |
  | -------- | ---------- | --: | ---: |
  | Europarl | Literature | 56% |  54% |

## 7. References

- Information Density and Quality Estimation Features as Translationese Indicators for Human Translation Classification (2016)  
  Raphaël Rubino, Ekaterina Lapshinova-Koltunski, Josef van Genabith. HLT-NAACL 2016

- Unsupervised Identification of Translationese (2015)  
  Ella Rabinovich, Shuly Wintner
  Transactions of the Association for Computational Linguistics, 3:419–432, 2015. ISSN 2307-387X

- Interpretese vs. Translationese:
  The Uniqueness of Human Strategies in Simultaneous Interpretation (2016)  
  He et al
  Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies

- Translationese and its dialects (2011)  
  Moshe Koppel and Noam Ordan, In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 1318–1326, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.

- Statistical Machine Translation
  with Automatic Identification of Translationese (2015)  
  Naama Twitto, Noam Ordan, Shuly Wintner
  Proceedings of the Tenth Workshop on Statistical Machine Translation

- Identification of Translationese: A Machine Learning Approach (2010)  
  Iustina Ilisei, Diana Inkpen, Gloria Corpas Pastor, Ruslan Mitkov
  CICLing 2010: Computational Linguistics and Intelligent Text Processing

- A New Approach to the Study of Translationese: Machine-learning the Difference between Original and Translated Text (2006)  
  Marco Baroni and Silvia Bernardini
  Literary and Linguistic Computing, 21(3): 259–274, September 2006

- A Parallel Corpus of Translationese (2016)  
  Ella Rabinovich, Shuly Wintner and Ofek Luis Lewinsohn
  Proceedings of the 17th International Confernece on Computational Linguistics and Intelligent Text Processing (CICLing-2016), pages 140-155, Konya, Turkey, April 2016.

- On the Features of Translationese (2015)  
  V Volansky, N Ordan, S Wintner  
  Literary and Linguistic Computing, Volume 30, Issue 1, 1 April 2015, Pages 98–118,
