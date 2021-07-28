import logging

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
# file_handler = logging.FileHandler('my.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

for i in range(10):
	logger.info(f'{i}번째 방문입니다.')
