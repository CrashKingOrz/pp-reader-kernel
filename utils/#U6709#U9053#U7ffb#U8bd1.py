import re

import requests
from lxml import etree


def parse2(html):
    # 正则表达式
    re_bds = '<div id="phrsListTab".*?>' \
             '.*?<div class="trans-container">' \
             '.*?<a class="search-js".*?">(.*?)</a>'

    # 生成正则表达式对象
    pattern = re.compile(re_bds, re.S)
    r_list = pattern.findall(html)
    r_list = list(r_list)

    if len(r_list) >= 1:

        html = crawl('http://youdao.com/w/eng/', r_list[0])  # 返回网页源代码
        result = parse(html)  # 解析网页 并保存
        result["word"] = r_list[0]
        return result
    else:
        return {}


def parse(html):
    result = {}

    tree = etree.HTML(html)
    # tree.xpath("//div[@class='phrsListTab']")
    examples = []
    result['英标'] = tree.xpath("//div[@id='phrsListTab']//div[@class='baav']//span[@class='phonetic']//text()")
    result['释义'] = tree.xpath("//div[@id='phrsListTab']//div[@class='trans-container']/ul/li/text()")
    example_list = tree.xpath("//div[@id='examples']//div[@id='bilingual']//ul[@class='ol']/li")
    for example in example_list:
        en = example.xpath("./p[1]//text()")
        ch = example.xpath("./p[2]//text()")
        examples.append({'英': "".join(en).strip(), '汉': "".join(ch).strip()})
    result['例子'] = examples

    return result

    # if "phrsListTab" in html:
    #
    #     # 正则表达式
    #     re_bds = '<div class="baav">' \
    #              '.*?<span class="phonetic">(.*?)</span>' \
    #              '.*?<span class="phonetic">(.*?)</span>' \
    #              '.*?<div class="trans-container">.*?<li>(.*?)</li>'
    #
    #     # 生成正则表达式对象
    #     pattern = re.compile(re_bds, re.S)
    #     # r_list: [('我不是药神','徐峥,周一围,王传君','2018-07-05'),...] 列表元组
    #     r_list = pattern.findall(html)
    #     r_list=list(r_list)
    #     if len(r_list)==0:
    #         print("error")
    #     else:
    #         print("英：{} 美：{} \n {}".format(r_list[0][0], r_list[0][1], r_list[0][2]))


def crawl(url, kw):
    # http://youdao.com/w/eng/scene/
    url = url + kw + '/'  # 构造url
    header = {
        'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        'accept-encoding': "gzip, deflate",
        'accept-language': "zh-CN,zh;q=0.9,en;q=0.8",
        'cache-control': "no-cache",
        'connection': "keep-alive",
        'cookie': "DICT_UGC=be3af0da19b5c5e6aa4e17bd8d90b28a|; OUTFOX_SEARCH_USER_ID=1768574849@220.181.76.83; webDict_HdAD=%7B%22req%22%3A%22http%3A//dict.youdao.com%22%2C%22width%22%3A960%2C%22height%22%3A240%2C%22showtime%22%3A5000%2C%22fadetime%22%3A500%2C%22notShowInterval%22%3A3%2C%22notShowInDays%22%3Afalse%2C%22lastShowDate%22%3A%22Mon%20Nov%2008%202010%22%7D; ___rl__test__cookies=1544516405027; JSESSIONID=abc3R7KYsXhP_9_VAwCEw; _ntes_nnid=a59fbbb6d3092682996becd378092a3d,1544516385701; OUTFOX_SEARCH_USER_ID_NCOO=928410427.493158; OUTFOX_SEARCH_USER_ID=\"486593599@10.169.0.83\"",
        'host': "youdao.com",
        'upgrade-insecure-requests': "1",
        'user-agent': "Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36"
    }
    response = requests.request('GET', headers=header, url=url)
    response.encoding = response.apparent_encoding
    return response.text


if __name__ == '__main__':
    mode = 2
    while True:

        word = input("please input a word:")
        if word == "q":
            break

        html = crawl('http://youdao.com/w/eng/', word)  # 返回网页源代码

        if mode == 2:
            result = parse(html)  # 解析网页 并保存
            print(result)
        else:
            result = parse2(html)  # 汉英
            print(result)
