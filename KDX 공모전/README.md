
# 소비트렌드 코리아 2020 - KDX한국데이터 거래소



## 개요
### 1. 상세설명
+ KDX에서 제공해준 유통/소비 금융사 데이터를 기반으로 온라인 상품 구매 데이터를 이용하여 전략적 온라인 광고 노출을 위한 OS유형별 소비패턴을 분석  
 - os유형별 나이대의 구매수와 금액을 비교 분석

### 2. 대회목표
+ KDX의 유통/소비 데이터와 다양한 외부 데이터를 활용해 한국의 유통&소비 특징을 파악할 수 있는 대시보드 구성 

### 3. 데이터셋 구성
+ 상품 대분류 카테고리 별로 분류
   - OS유형
   - 성별
   - 나이
   - 구매수
   - 구매금액 데이터

### 4. 상금 (총 600만원)
+ 대상(1팀) : 300만원
+ 최우수상(1팀) : 150만원
+ 우수상(1팀) : 100만원
+ 장려상(2팀) : 25만원

### 5. 규칙
+ [분석환경 내 제공 데이터]
  - MBN 뉴스데이터 / 엠코퍼레이션 온라인 구매 데이터 /  삼성카드 구매 데이터
    /신한카드 오프라인 구매 데이터 / 지인플러스 전국 아파트 시세 및 거래량 데이터 외 

+ [이용가능한 데이터]
  - KDX 무료 데이터 / 외부 또는 공공 데이터 / 크롤링 데이터 외 외부 데이터는 공공 데이터와
    같이 법적인 제약이 없는 경우에만 사용 가능하며 주석을 통한 데이터 설명 및 링크 제시
  - 크롤링 데이터 사용 시 크롤링 코드 제출 필수 
  - 머신 러닝, 상관관계 분석, 회귀분석 등 고급 통계 분석 기법을 쓸 경우 가산점  부여
  - 시각화 차트는 필수적으로 5개 이상을 써야 하며, 창의적이고 설명력 있는 인사이트 제시할 경우 가산점 부여
  - 대시보드 구성의 참신성과 스토리텔링 능력에 따라 가산점 부여
  
+ 상세 규칙
 - 팀은 개인 또는 최대 4인까지 구성 가능
 
 ### 6. 참가
* 작업툴 : Python Jupyter Notebook, R studio
* 인원 : 3명
* 기간 : 2020.10.10 ~ 2020.10.24
* 내용 : 공모전에서 주어진 카드사용내역서 파일을 참조하여 전략적 온라인 광고 노출을 위한 OS유형별 소비패턴을 분석한 프로젝트



## 프로그램 소스코드 설명
### 1. 라이브러리 불러오기
```{r}
library(tidyverse) 
library(readxl) 
library(ggplot2)
library(lubridate)
library(dplyr)
library(zoo)
library(modelr)
options(na.action = na.warn)
library(pastecs)
library(WRS)
library(utils)
```
* tidyverse, dplyr와 ggplot2는 데이터를 가공하고 시각화하는데 사용하였습니다.
* readxl은 엑셀파일을 불러오기 위하여 사용하였습니다.
* lubridate 패키지는 날짜를 가져오기 위하여 사용하였습니다.
* zoo, modelr, pastecs, WRS, untils 패키지는 기술통계량과 t검정을 하기 위하여 사용하였습니다.





### 2. 데이터 불러오기 및 데이터 통합
```
files <- list.files(path = "data/Mcorporation/상품 카테고리 데이터_KDX 시각화 경진대회 Only/", pattern = "*.xlsx", full.names = TRUE) # 다중 엑셀파일 불러오기

glimpse(files)

products <- sapply(files[2:65], read_excel, simplify=FALSE) %>% 
  bind_rows(.id = "id") %>% 
  select(-id)

glimpse(products)
```

### 3. 기술통계량
#### 3-1) 고객성별 요약
```
products %>%
  filter(고객성별 != "없음") %>%
  select(고객성별, 구매금액, 구매수) %>%
  group_by(고객성별) %>%
  summarize(구매금액평균 = mean(구매금액), 구매수평균 = mean(구매수)) %>%
  mutate(금액비율 = 구매금액평균 * 100 / sum(구매금액평균), 수량비율 = 구매수평균 * 100 / sum(구매수평균))
  
#### 3-2) OS유형별 요약
products %>%
  filter(OS유형 != "없음") %>%
  select(OS유형, 구매금액, 구매수) %>%
  group_by(OS유형) %>%
  summarize(구매금액평균 = mean(구매금액), 구매수평균 = mean(구매수)) %>%
  mutate(금액비율 = 구매금액평균 * 100 / sum(구매금액평균), 수량비율 = 구매수평균 * 100 / sum(구매수평균))

#### 3-3) 고객나이별 요약
products %>%
  filter(고객나이 > 0 & 고객나이 < 80) %>%
  select(고객나이, 구매금액, 구매수) %>%
  mutate(금액비율 = 구매금액 * 100 / sum(구매금액), 수량비율 = 구매수 * 100 / sum(구매수)) %>%
  group_by(고객나이) %>%
  summarize(구매금액평균 = mean(구매금액), 구매수평균 = mean(구매수), 금액비 = sum(금액비율), 수량비 = sum(수량비율))
  
#### 3-4) 카테고리별 요약
products %>%
  select(카테고리명, 구매금액, 구매수) %>%
  group_by(카테고리명) %>%
  summarize(구매금액평균 = mean(구매금액)) %>%
  mutate(금액비율 = 구매금액평균 * 100 / sum(구매금액평균)) %>%
  arrange(desc(구매금액평균)) %>%
  head(10)
  
#### 3-5) 구매금액 총합
products %>%
  select(카테고리명, 구매금액, 구매수) %>%
  group_by(카테고리명) %>%
  summarize(구매금액평균 = mean(구매금액)) %>%
  summarize(평균합 = sum(구매금액평균))


products %>%
  select(카테고리명, 구매금액, 구매수) %>%
  group_by(카테고리명) %>%
  summarize(구매수평균 = mean(구매수)) %>%
  mutate(금매수비율 = 구매수평균 * 100 / sum(구매수평균)) %>%
  arrange(desc(구매수평균)) %>%
  head(10)

#### 3-6) 구매수 평균 총합
products %>%
  select(카테고리명, 구매금액, 구매수) %>%
  group_by(카테고리명) %>%
  summarize(구매수평균 = mean(구매수)) %>%
  summarize(평균합 = sum(구매수평균))
```

### 4. OS별 매달 구매금액 변화 그래프
```
products2 <- products %>% 
  mutate(년월 = as.yearmon(as.character(구매날짜), "%Y%m")) %>%
  rename(date = "년월") %>%
  filter(OS유형 != "없음") %>%
  group_by(date, OS유형) %>%
  summarise(mean = mean(구매금액))

products_line <- products2 %>% 
  data_grid(date, OS유형) %>%
  gather_predictions(lm(mean ~ date * OS유형, data = products2))

products2 %>%
  ggplot(aes(date, mean, colour = OS유형)) +
  geom_line(lwd = 2) +
  geom_line(data = products_line, aes(y = pred),lwd = 1) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw() +
  theme(legend.title = element_text(size = 15, face = "bold")) +
  theme(legend.text = element_text(size = 12))
```

### 5. 안드로이드의 카테고리별 구매금액
```
products %>%
  filter(OS유형 == "안드로이드") %>%
  group_by(카테고리명, OS유형) %>%
  summarise(mean = mean(구매금액)) %>%
  arrange(desc(mean)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(카테고리명, -mean), y = mean)) +
  geom_bar(stat="identity", position = "dodge", width=.5, fill = "#619CFF") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6,)) + 
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw()
```

### 6. IOS의 카테고리별 구매금액
```
products %>%
  filter(OS유형 == "IOS") %>%
  group_by(카테고리명, OS유형) %>%
  summarise(mean = mean(구매금액)) %>%
  arrange(desc(mean)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(카테고리명, -mean), y = mean)) +
  geom_bar(stat="identity", position = "dodge", width=.5, fill = "#fc9272") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6,)) + 
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw()
```

### 7. 안드로이드의 나이대별 남녀 소비 비율
```
products %>%
  filter(OS유형 == "안드로이드") %>%
  filter(고객나이 > 0 & 고객나이 < 80) %>%
  group_by(고객나이, 고객성별) %>%
  summarise(mean = mean(구매금액)) %>%
  ggplot(aes(x = 고객나이, y = mean, fill = 고객성별)) +
  geom_bar(stat="identity", position = "dodge", width=5) + 
  scale_x_continuous(breaks = seq(0, 80, by = 10)) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw()

```

### 8. IOS의 나이대별 남녀 소비 비율
```
products %>%
  filter(OS유형 == "IOS") %>%
  filter(고객나이 > 0 & 고객나이 < 80) %>%
  group_by(고객나이, 고객성별) %>%
  summarise(mean = mean(구매금액)) %>%
  ggplot(aes(x = 고객나이, y = mean, fill = 고객성별)) +
  geom_bar(stat="identity", position = "dodge", width=5) + 
  scale_x_continuous(breaks = seq(0, 80, by = 10)) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw()
```

### 9. 안드로이드와 IOS 나이대별 구매금액
```
products %>%
  filter(OS유형 == "안드로이드" | OS유형 == "IOS") %>%
  filter(고객나이 > 0 & 고객나이 < 80) %>%
  group_by(고객나이, OS유형) %>%
  summarise(mean = mean(구매금액)) %>%
  ggplot(aes(x = 고객나이, y = mean, fill = OS유형)) +
  geom_bar(stat="identity", position = "dodge", width=5) + 
  scale_fill_manual(values = c('#F8766D','#619CFF')) +
  scale_x_continuous(breaks = seq(0, 80, by = 10)) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_bw()
```

### 10. OS별 매달 구매금액 변화 그래프
```
products %>% 
  mutate(년월 = as.yearmon(as.character(구매날짜), "%Y%m")) %>%
  rename(date = "년월") %>%
  filter(OS유형 != "없음") %>%
  group_by(date, OS유형) %>%
  summarise(mean = mean(구매금액)) %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = mean, colour = OS유형),lwd = 2) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme(legend.title = element_text(size = 15, face = "bold")) +
  theme(legend.text = element_text(size = 12)) +
  theme_bw()

```

### 11. OS별 매달 구매수 변화 그래프
```
products %>% 
  mutate(년월 = as.yearmon(as.character(구매날짜), "%Y%m")) %>%
  rename(date = "년월") %>%
  filter(OS유형 != "없음") %>%
  group_by(date, OS유형) %>%
  summarise(mean = mean(구매수)) %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = mean, colour = OS유형),lwd = 2) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme(legend.title = element_text(size = 15, face = "bold")) +
  theme(legend.text = element_text(size = 12)) + 
  theme_bw()
```


### 12. 독립 t검정
#### 12-1) 데이터 필터링
```
products1 <- products %>%
  mutate(년월 = as.yearmon(as.character(구매날짜), "%Y%m")) %>%
  rename(date = "년월") %>%
  filter(OS유형 == "IOS" | OS유형 == "안드로이드") %>%
  group_by(date, OS유형) %>%
  summarise(mean = mean(구매금액))

glimpse(products1)
```

#### 12-2) 평균과 표준오차
```
by(products1$mean, products1$OS유형, stat.desc, basic = FALSE, norm = TRUE)
```

#### 12-3) t.test()
```
ind.t.test<-t.test(mean ~ OS유형, data = products1)

ind.t.test
```

