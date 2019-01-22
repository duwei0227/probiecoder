---
layout: page
title: About
description: 一个码农的修仙之旅
keywords: Du Wei, Tony Du, 杜伟
comments: true
menu: 关于
permalink: /about/
---

一个码农的修仙之旅

## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：<a href="{{website.url}}" target="_blank">@{{ website.name }}</a>
{% endfor %}

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
