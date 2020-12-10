
# 소설 작가 분류 AI 경진대회 -DACON



## 개요
### 1. 상세설명
+ 작가의 글을 분석하여 특징 도출
+ 취향 추천 시스템 활용 / 대필, 유사작 탐지

### 2. 대회목표
+ 글을 분석하여 작가의 특징을 도출해 내며, 취향 추천과 대필, 혹은 유사작을 탐지해 내는것이 이 대회의 목표이다.

### 3. 상금 (총 100만원 + 애플워치)
+ 1등 : 이노베이션 아카데미 "42 Prize": 애플 워치 SE
+ 2~11등 : ₩ 100,000

### 5. 규칙
+ 1. 평가
    + 심사 기준: LogLoss
    + 1차 평가(Public Score): 테스트 데이터 중 랜덤 샘플 된 30%로 채점, 대회 기간 중 공개
    + 2차 평가(Private Score): 나머지 70 % 테스트 데이터로 채점, 대회 종료 직후 공개
    + 최종 순위는 선택된 파일 중에서 채점되므로, 참가자는 제출 창에서 자신이 최종적으로 채점 받고 싶은 파일을 선택해야 함. (최종 파일 미선택시 처음으로 제출한 파일로 자동 선택됨)
    + 대회 직후 공개되는 Private Score 랭킹은 최종 순위가 아니며, 코드 검증 후 최종 수상자가 결정됨




+ 2. 개인 또는 팀 참여 규칙
    + 개인 또는 팀을 이루어 참여할 수 있습니다.
    + 단체 혹은 기관 참여시 별도의 절차가 필요합니다. (More > 공지사항> 게시글 확인)
    + 개인 참가 방법: 팀 신청 없이, 자유롭게 제출 창에서 제출 가능
팀 구성 방법: 팀 페이지(https://www.dacon.io/competitions/official/235670/team/)에서 팀 구성 안내 확인
    + 팀 최대 인원: 3 명
                   * 동일인이 개인 또는 복수팀에 중복하여 등록 불가.

 

+ 3. 외부 데이터 및 사전 학습 모델
    + 외부 데이터 사용이 불가합니다. 
    + 사전 학습 모델(pre-trained Model) 사용이 불가합니다.




+ 4. 유저평가
    + DACON Scholarship을 받고자 하는 팀은 유저 평가를 받아야 합니다.
    + Private 순위 공개 후 코드 제출 기간 내 코드 공유 페이지에 코드 업로드
    + 제목에 Private 순위와 Public 점수를 기입
예시) Private 1위, Public 점수 :0.98, LGBM 모델
    + 대회 참가자는 공개된 코드 평가
    + 코드 오류, 외부 데이터 사용 등 코멘트를 댓글로 작성




+ 5. 유의 사항
    + 1일 최대 제출 횟수: 3회
    + 사용 가능 언어: Python, R
    + 모델 학습에서 검증 혹은 평가 데이터셋 활용(Data Leakage)시 실격
    + 최종 순위는 선택된 파일 중에서 채점되므로 참가자는 제출 창에서 자신이 최종적으로 채점 받고 싶은 파일을 선택해야 함
    + 대회 직후 공개되는 Private 랭킹은 최종 순위가 아니며 코드 검증 후 수상자가 결정됨
    + 데이콘은 부정 제출 행위를 금지하고 있으며 데이콘 대회 부정 제출 이력이 있는 경우 평가가 제한됩니다. 자세한 사항은 아래의 링크를 참고해 주시기 바랍니다. https://dacon.io/notice/notice/13


## 프로그램 소스코드 설명
### Library

## * sentencepiece는 직접 설치가 필요합니다 *
### py파일을 실행하기 전에 pip install sentencepiece를 입력해 주세요


```python
pip install tensorflow
```


```python
from pandas import read_csv
import re
from sentencepiece import SentencePieceTrainer,SentencePieceProcessor
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from keras.models import load_model
from numpy import mean
from pandas import DataFrame
from random import randint

# 코드에 필요한 파일 : data, models, submissions
# import os
# os.mkdir('/content/drive/MyDrive/Dacon-소설작가분류/models')
# os.mkdir('/content/drive/MyDrive/Dacon-소설작가분류/submissions)
```

# Data Loading
## 1. file에서 data 불러오기
## 2. Sentence Piece로 모델 학습
## 3. CountVectorizer에 tokenizer로 trained model 사용
## 4. 학습용 데이터 최종 Loading


```python
def data_loading():
  
  # 저는 data라는 폴더 안에 데이터를 저장해 두었습니다.

  train = read_csv('/content/drive/MyDrive/Dacon-소설작가분류/data/train.csv', engine='python')
  test = read_csv('/content/drive/MyDrive/Dacon-소설작가분류/data/test_x.csv', engine='python')
  y_train = train['author'].values
  print('data load completed')


  # 필요없는 단어 제거
  def alpha_num(text):
      return re.sub(r'[^a-zA-z0-9\s]', '', text)

  # 소문자로 변환
  train['text'] = train['text'].str.lower().apply(alpha_num)
  test['text'] = test['text'].str.lower().apply(alpha_num)
  print('data transformation completed')
  del alpha_num

  # SentencePieceTrainer로 학습할 데이터 준비
  with open('author.txt','w',encoding='utf-8') as f:
      f.write('\n'.join(train['text']))
  del f

  # 3000자까지 vocab size를 지정해서 학습
  SentencePieceTrainer.Train('--input=author.txt --model_prefix=author --vocab_size=3000')
  
  sp = SentencePieceProcessor()
  sp.Load("author.model")

  # 학습된 모델을 tokenizer로 사용 / min_df = 3은 최소 3번 이상 나타난 단어를 의미합니다
  cv = CountVectorizer(lowercase = False, tokenizer = sp.encode_as_pieces, min_df = 3)

  # 최종적으로 사용할 학습데이터입니다.
  tdm_train = cv.fit_transform(train['text']).toarray()
  del train
  tdm_test = cv.transform(test['text']).toarray()
  print('data setting completed')

  return tdm_train, tdm_test, y_train

```

# Simple DNN Code
## Input Shape는 vocabsize = 3000을 기준으로 약 3000개의 단어가 들어갑니다.


```python
# 간단한 DNN 모델
def create_dnn(input_shape):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape = (input_shape,)))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam())
    return model

```

# Creating CSV
## 1. StratifiedKFold로 5 fold
## 2. val_loss를 기준으로 Early Stopping &  Model Checkpoint monitoring
## 3. Checked Model 불러와 tdm_test 예측 + predicts에 저장
## 4. log_loss로 Validation data score 저장
## 5. 5 Fold의 predicts의 mean => csv로 저장


```python
# csv를 만드는 코드
def creating_results(seed, tdm_train, tdm_test, y_train):
  batch_size =  256
  epochs = 30

  # SKF 중 생성되는 결과물 저장 
  predicts = []
  # Validation Score 저장
  scores = []
  
  
  # 과적합이 쉽게 되어 Patience는 2
  es = EarlyStopping(monitor='val_loss', verbose=0, patience=2)

  # 학습하며 생되는 model 저장
  filepath_val_loss="/content/drive/MyDrive/Dacon-소설작가분류/models/best_model_cdnn.tf"
  checkpoint_val_loss = ModelCheckpoint(filepath_val_loss, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

  # SKF
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  for train_index, val_index in skf.split(tdm_train, y_train):
    model = create_dnn(tdm_train.shape[1])
    tr_X = tdm_train[train_index]
    tr_y = y_train[train_index]

    val_X = tdm_train[val_index]
    val_y = y_train[val_index]

    model.fit(tr_X,tr_y,
                        batch_size=batch_size,
                        epochs=epochs, 
                        validation_data=(val_X,val_y), 
                        shuffle=True, 
                        verbose=0,
                        callbacks=[es,checkpoint_val_loss]
                        ) 
    
    model = load_model(filepath_val_loss)
    predicts.append(model.predict_proba(tdm_test))
    score = log_loss(val_y,model.predict_proba(val_X))
    scores.append(score)
    del tr_X,tr_y,val_X,val_y,model,score
  
  # predicts에 저장된 predicts 저장
  prediction = DataFrame(mean(predicts,axis=0)).reset_index()

  # 대략적으로 어느정도의 validation loss를 가지고있는지 파악하기 위해 점수도 함께 저장
  score = mean(scores)
  print(score)
  file_path = '/content/drive/MyDrive/Dacon-소설작가분류/submissions/SIMPLE_DNN_'+str(score)[:8]+'.csv'
  prediction.to_csv(file_path, index=False) 
  print('#'*25,"FINISHED  : ",score, ' '*10,'#'*25)
  del prediction, score, file_path, filepath_val_loss,es,skf,train_index,val_index,scores,predicts,epochs,batch_size,

# seed는 random으로 지정되며, 결과물이 생성됩니다.
def main():
  seed = randint(1,1000000)
  tdm_train, tdm_test, y_train = data_loading()
  creating_results(seed, tdm_train, tdm_test, y_train)
```


```python
if __name__ == "__main__":
  main()
```

# py파일을 실행할 때는, 새로운 ipynb를 생성하여
- pip install sentencepiece
- !python3 'filepath/simple_dnn.py' 를 실행합니다.

# 2020년 11월 18일 12시 30분에 생성 결과 : 0.4288

#### 다만, 제목과 같이 SIMPLE한 모델입니다. 
#### 좀 더 tuning하고 regulize한다면 더 좋은 결과를 얻을 수 있을 것이라고 생각합니다.



```python

```



