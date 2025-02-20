# 파이썬 표준 라이브러리
import json
import time
from pathlib import Path

# 파이썬 서드파티 라이브러리
from news_collect import RealTimeCrawling


if __name__ == '__main__':
    rtc = RealTimeCrawling()
    website_list = ['hankyung', 'maekyung', 'yna']
    for website in website_list:
        rtc.add_website(website)

    while True:
        json_path = rf'{Path(__file__).parents[0]}/config.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            process = json.load(f)['PROCESS_VARIABLE']

        if process == "True":
            rtc.run()
            time.sleep(60)
        else:
            # config.json 파일의 PROCESS_VARIABLE 값을 True로 바꾸고 종료
            json_path = rf'{Path(__file__).parents[0]}/config.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({'PROCESS_VARIABLE': 'True'}, f)
            break