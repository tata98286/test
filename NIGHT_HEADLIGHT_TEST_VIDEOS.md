# 야간 헤드라이트·반사광 비화재 테스트 영상

확인일: 2026-09-23

화재 모델이 헤드라이트, 가로등, 젖은 노면 반사, 안개 속 산란광을 불꽃·연기로 잘못 판단하는지 확인하기 위한 후보 목록이다. 모두 현재 웹의 YouTube 주소 입력 제한인 30분 이내다.

## A. 정면 헤드라이트 번짐

| 번호 | 길이 | 영상 | 테스트 목적 |
|---:|---:|---|---|
| A1 | 1:01 | [Night Vision - Front Camera 4K - City Headlight Glare](https://www.youtube.com/watch?v=x4WlWU-CKmg) | 정면 헤드라이트와 도심 조명 번짐 |
| A2 | 1:01 | [Headlight Glare & Plate Readability Dark Road](https://www.youtube.com/watch?v=obtYGSIaw3M) | 배경이 어두운 도로의 강한 점광원 |
| A3 | 1:01 | [Headlight Glare & Plate Readability City](https://www.youtube.com/watch?v=KBXhHerCyRM) | 여러 차량 불빛과 시내 반사광 |
| A4 | 0:21 | [Headlights, streetlights moving towards camera in darkness](https://www.youtube.com/watch?v=HikQ5mdamJQ) | 카메라로 다가오는 불빛과 가로등 |

## B. 후면 카메라·유리 반사

| 번호 | 길이 | 영상 | 테스트 목적 |
|---:|---:|---|---|
| B1 | 1:00 | [Rear Camera - Headlight Glare & Plate Visibility](https://www.youtube.com/watch?v=QtcC2hWbEoA) | 후방 차량 헤드라이트 번짐 |
| B2 | 1:01 | [Rear Camera - Headlight Glare & Plate Visibility 2](https://www.youtube.com/watch?v=k3gtJG1QQmM) | 같은 장면 유형의 반복성 확인 |
| B3 | 1:01 | [Dirty Rear Glass Headlight Glare](https://www.youtube.com/watch?v=rUyH8y3qnCU) | 더러운 유리에 퍼지는 빛 |
| B4 | 1:01 | [Rear Camera - Rainy Midnight](https://www.youtube.com/watch?v=ZzfMqk-DH0I) | 비·물방울과 헤드라이트 산란 |

## C. 비·젖은 노면·도심 반사광

| 번호 | 길이 | 영상 | 테스트 목적 |
|---:|---:|---|---|
| C1 | 8:19 | [Driving Through a Nighttime Rainstorm](https://www.youtube.com/watch?v=UBDGeaTF_wc) | 젖은 노면의 넓은 반사 영역 |
| C2 | 2:27 | [Highway driving at night over the bridge](https://www.youtube.com/watch?v=fo3WjGm0bTQ) | 다수 차량·교량 조명이 섞인 장면 |
| C3 | 2:40 | [Cars driving at night](https://www.youtube.com/watch?v=Ina7KMV2OEI) | 일반 야간 교통 기준 영상 |
| C4 | 0:59 | [Traffic At Night, Speed, Car, Headlights, City](https://www.youtube.com/watch?v=QeRKe_Qe9kA) | 이동하는 헤드라이트 궤적과 도시 조명 |

## D. 안개·연기 유사 장면

| 번호 | 길이 | 영상 | 테스트 목적 |
|---:|---:|---|---|
| D1 | 1:01 | [Night countryside in fog](https://www.youtube.com/watch?v=tvu7RlsE5jw) | 안개를 연기로 오인하는지 확인 |
| D2 | 1:01 | [Night foggy countryside](https://www.youtube.com/watch?v=nux0AtiovFw) | 안개와 헤드라이트의 결합 |
| D3 | 1:01 | [Night countryside dark](https://www.youtube.com/watch?v=SWSOpV-3nGM) | 가로등이 거의 없는 낮은 조도 |
| D4 | 0:30 | [Night - No street lighting](https://www.youtube.com/watch?v=YSjE9YjzuR8) | 작은 점광원에 대한 민감도 |

## 권장 실험 순서

1. A1, B3, C1, D1을 먼저 검사해 네 가지 혼동 유형을 빠르게 확인한다.
2. 각 이벤트에서 사람 판정을 **비화재/오탐**으로 저장하고 실제로 화재 장면이 없는지 사람이 원본을 다시 확인한다.
3. 같은 자료를 OR와 AND로 각각 검사한다.
4. YOLO 후보 수, DINO 통과·차단 수, VLM NO 비율, 시간당 오탐 수를 비교한다.
5. 오류가 많이 난 유형만 같은 분류의 나머지 영상으로 확대 검사한다.

## 기록할 정답

| 입력 자료 | 사람 정답 | 확인할 오인 유형 |
|---|---|---|
| 헤드라이트 영상 | 비화재 | 불꽃 오인 |
| 비·젖은 노면 | 비화재 | 반사광 오인 |
| 안개 영상 | 비화재 | 연기 오인 |
| 실제 화재 대조군 | 실제 화재 | 오탐 감소 과정에서 미탐 증가 여부 |

링크가 삭제되거나 비공개로 바뀔 수 있다. 제목만 보고 정답을 확정하지 말고, 검사 전후에 사람이 전체 영상을 확인해 실제 화재가 없는 자료만 비화재 정답으로 사용한다.
