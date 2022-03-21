# -*- ecoding: utf-8 -*-
# @ModuleName: 百度汉语
# @Function: 
# @Author: qy
# @Time: 2022/2/10 15:33

# -*- ecoding: utf-8 -*-
# @ModuleName: 汉语词典
# @Function:
# @Author: qy
# @Time: 2022/2/10 15:28
"""一：百度词语爬虫"""
import urllib.request
from lxml import etree
from urllib.parse import urlencode, unquote
import requests
import re
import json
import time

def digui(url,headers):
    try:
        request = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(request, timeout=0.7).read().decode("utf8")
        return html
    except:
        time.sleep(10)
        digui(url,headers)


def load_baidu_page(kw,url):
    # 获取html页面
    dic1 = {}
    """获取html页面"""
    headers = {
        'Accept': 'text / html, application / xhtml + xml, application / xml,*/*;q = 0.9;q = 0.8',
        'Accept - Encoding': 'gzip, deflate, br',
        'Accept - Language': 'zh - CN, zh;q = 0.9',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"

    }
    html = digui(url,headers)
    content = etree.HTML(str(html))
    if len(kw) == 1:
        link_list_pinyin = content.xpath('//div[@class="pronounce"]//b/text()')     #拼音
        link_list0=content.xpath('//div[@class="content means imeans"][1]//p/text()')            #基本释义
        link_list1 = content.xpath('//div[@class="content means imeans"][2]//p/text()')            #详细释义
        link_source=content.xpath('//div[@id="source-wrapper"]//p/text()')            #出处
        link_liju=content.xpath('//div[@id="liju-wrapper"]//p/text()')            #例句
        link_story=content.xpath('//div[@id="story-wrapper"]//p/text()')            #典故
        link_synonym = content.xpath('//div[@id="synonym"]//a/text()')  # 近义词
        link_antonym = content.xpath('//div[@id="antonym"]//a/text()')  # 反义词
        link_redical = content.xpath('//li[@id="radical"]/span/text()')        #部首
        link_stroke = content.xpath('//li[@id="stroke_count"]/span/text()')        #笔画
        link_content = content.xpath('//div[@class="tab-content"]/a/text()')         #相关组词

        dic1["关键词"] = kw
        dic1['拼音'] = link_list_pinyin
        dic1["基本释义"] = "".join(link_list0).strip()
        dic1["详细释义"] = "".join(link_list1).strip()
        dic1["近义词"] = link_synonym
        dic1["反义词"] = link_antonym
        dic1["部首"] = link_redical
        dic1["笔画"] = link_stroke
        dic1["相关组词"] = link_content
    else:
        #获取详细信息
        link_list_pinyin = content.xpath('//div/dl/dt[@class="pinyin"]/text()')     #拼音
        link_list0=content.xpath('//div[@class="content means imeans"][1]//p/text()')            #基本释义
        link_list1 = content.xpath('//div[@class="content means imeans"][2]//p/text()')            #详细释义
        link_source=content.xpath('//div[@id="source-wrapper"]//p/text()')            #出处
        link_liju=content.xpath('//div[@id="liju-wrapper"]//p/text()')            #例句
        link_story=content.xpath('//div[@id="story-wrapper"]//p/text()')            #典故
        link_synonym = content.xpath('//div[@id="synonym"]//a/text()')  # 近义词
        link_antonym = content.xpath('//div[@id="antonym"]//a/text()')  # 反义词
        link_content = content.xpath('//div[@class="tab-content"]/a/text()')         #相关组词

        dic1["关键词"] = kw
        dic1['拼音'] = link_list_pinyin
        dic1["基本释义"] = "".join(link_list0).strip()
        dic1["详细释义"] = "".join(link_list1).strip()
        dic1["例句"] = link_liju
        dic1["出处"] = link_source
        dic1["典故"] = link_story
        dic1["近义词"] = link_synonym
        dic1["反义词"] = link_antonym
        dic1["相关组词"] = link_content

    # for key in dic1:
    #     print("{}:{}".format(key,dic1[key]))
    a=dic1['拼音']
    print(len(dic1['拼音']))

def baidu_hanyu(kw):
    url = "https://hanyu.baidu.com/s"
    word = {"wd": kw}
    key = urllib.parse.urlencode(word)
    fullurl = url + "?" + key + "&ptype=zici"  # 完整url
    load_baidu_page(kw, fullurl)  # 进入到百度搜索结果的页面

if __name__ == "__main__":
    """输入要搜索的关键词和对应的url地址"""
    while(True):
        kw = input("请输入要搜索的关键词： ")
        baidu_hanyu(kw)


