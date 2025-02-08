# -*- coding: utf-8 -*-


# 사용자 정의 라이브러리
from crawling_scraping_core import crawling_scraping



if __name__ == '__main__':
    cs = crawling_scraping.CrawlingScraping(record_log=True)
    end_datetime, date_format = '2024-01-01 00:00', '%Y-%m-%d %H:%M'
    website_list = ('hankyung', 'maekyung', 'yna',)
    for website in website_list:
        cs.add_website(website=website)

    cs.run(end_datetime=end_datetime, date_format=date_format)
    cs.to_json()