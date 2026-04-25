import requests
from lxml import etree #提取数据
url =
headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome' }
res = requests.get(url, headers=headers)
html = etree.HTML(res.content.decode())
imgs = html.xpath('//*[@id="js-live-list"]//li[*]/a[1]/img/@data-original')
titles = html.xpath('//*[@id="js-live-list"]//li[*]/a[1]/img/@title')
for i in range(len(imgs)):  # 遍历所有图片
    res2 = requests.get(imgs[i], headers=headers)  # 下载图片
with open('./mm/%s.jpg' % titles[i], 'wb') as f:
    print(imgs[i])  # 打印图片URL
    print(titles[i])  # 打印图片标题
    f.write(res2.content)  # 保存图片内容

