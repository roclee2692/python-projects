import os
import re
import requests
from lxml import etree

url ="https://v2.kwaicdn.com/ksc2/2ib5K9rDnmwhJ2sV2fhnA1hL1HKJYj6-wSOzC0Lbg5xiYawdfAsLTWsaup5K4Hb2OJ3FdZOQ0zqObDcArBQS5Nk9KXWpo81R3FuIM4WvNa7Ruo2ayJM6TkjFvXXtcWqt.mp4?pkey=AAXGWAtkBywawtTrj9J2xX1GzJH4Rup3u0SjLaPLKyf7u-VeBGyf68tTJDEMAo7QXSZo1ZizYFP_cyq9NvcO0kKUUKlsfw2AjRFSBS5ujaV5JZw0QHvAemZvqW5lnXh9_U4&tag=1-1741065229-unknown-0-4bvvyfqjrf-1bd7c8913b61e243&clientCacheKey=3x4sbrtqm3k9fd9_52c8051f&di=JAmJSYKQ7QzlbRVUt_bmqw==&bp=14944&tt=hd15&ss=vp"  # 替换为目标网页地址
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome'}

res = requests.get(url, headers=headers)
html = etree.HTML(res.content.decode())

imgs = html.xpath('//*[@id="js-live-list"]//li[*]/a[1]/img/@data-original')
titles = html.xpath('//*[@id="js-live-list"]//li[*]/a[1]/img/@title')

# 创建保存目录
if not os.path.exists('./mm'):
    os.makedirs('./mm')

for i in range(len(imgs)):
    try:
        res2 = requests.get(imgs[i], headers=headers, timeout=10)
        res2.raise_for_status()
    except requests.RequestException as e:
        print(f"下载失败: {e}")
        continue

    # 处理非法文件名
    title_clean = re.sub(r'[\/:*?"<>|]', '_', titles[i])

    with open(f'./mm/{title_clean}.jpg', 'wb') as f:
        print(f"正在下载: {imgs[i]}")
        print(f"保存为: {title_clean}.jpg")
        f.write(res2.content)
