import requests
url="https://cn-lnsy-cm-01-02.bilivideo.com/upgcxcode/10/74/59477410/59477410_nb2-1-16.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1741098651&gen=playurlv2&os=bcache&oi=663794702&trid=0000a48838df244d4a66bff221343b10176dh&mid=0&platform=html5&og=cos&upsig=230dc8520ebf16ec46f59125bd08810a&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&cdnid=3243&bvc=vod&nettype=0&f=h_0_0&bw=46085&logo=80000000"
#url = "https://xy139x170x157x110xy2408y8776y1y33yy34xy.mcdn.bilivideo.cn:4483/upgcxcode/16/63/28609416316/28609416316-1-100026.m4s?e=ig8euxZM2rNcNbdlhoNvNC8BqJIzNbfqXBvEqxTEto8BTrNvN0GvT90W5JZMkX_YN0MvXg8gNEV4NC8xNEV4N03eN0B5tZlqNxTEto8BTrNvNeZVuJ10Kj_g2UB02J0mN0B5tZlqNCNEto8BTrNvNC7MTX502C8f2jmMQJ6mqF2fka1mqx6gqj0eN0B599M=&uipk=5&nbs=1&deadline=1741096690&gen=playurlv2&os=mcdn&oi=663794702&trid=00000865c2d18d0442c1bc59ab72ad9e46e7u&mid=519251443&platform=pc&og=cos&upsig=069993fb5477ae51b8441ecf54ce1aca&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&mcdnid=50015020&bvc=vod&nettype=0&orderid=0,3&buvid=D314E2C0-8D35-24B7-6554-B555390C786692076infoc&build=0&f=u_0_0&agrr=0&bw=229570&logo=A002000"
headers = {"User-Agent": "Mozilla/5.0"}
res = requests.get(url, headers=headers, stream=True)
print(res.status_code)
with open("video3.mp4", "wb") as f:
    for chunk in res.iter_content(chunk_size=1024):
        f.write(chunk)

print("视频下载完成")
