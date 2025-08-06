url="https://xy139x170x157x110xy2408y8776y1y33yy34xy.mcdn.bilivideo.cn:4483/upgcxcode/16/63/28609416316/28609416316-1-100026.m4s?e=ig8euxZM2rNcNbdlhoNvNC8BqJIzNbfqXBvEqxTEto8BTrNvN0GvT90W5JZMkX_YN0MvXg8gNEV4NC8xNEV4N03eN0B5tZlqNxTEto8BTrNvNeZVuJ10Kj_g2UB02J0mN0B5tZlqNCNEto8BTrNvNC7MTX502C8f2jmMQJ6mqF2fka1mqx6gqj0eN0B599M=&uipk=5&nbs=1&deadline=1741096690&gen=playurlv2&os=mcdn&oi=663794702&trid=00000865c2d18d0442c1bc59ab72ad9e46e7u&mid=519251443&platform=pc&og=cos&upsig=069993fb5477ae51b8441ecf54ce1aca&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&mcdnid=50015020&bvc=vod&nettype=0&orderid=0,3&buvid=D314E2C0-8D35-24B7-6554-B555390C786692076infoc&build=0&f=u_0_0&agrr=0&bw=229570&logo=A002000"
import requests
headers={'user-agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1 Edg/133.0.0.0'}
res=requests.get(url, headers=headers)
print(res.status_code)#响应码 200 successful 403 408 500
open("agabfbagf.mp4","wb").write(res.content)

