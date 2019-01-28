---
layout: post
title: 支持jupyter远程访问
categories: [Python, Jupyter]
description: 支持jupyter远程访问
keywords: jupyter, 远程访问
---

支持jupyter远程访问

#### 1、生成jupyter config文件
`jupyter notebook --generate-config`  
文件默认生成路径为：
`~/.jupyter/jupyter_notebook_config.py`  

#### 2、生成密码，配置文件中需要
```Python
ipython
from notebook.auth import passwd
passwd()
Enter password: 
Verify password:
```

#### 3、编辑第一步生成的config文件
修改如下内容：
```
c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #可自行指定一个端口, 访问时使用该端口
```

参考：[https://blog.csdn.net/simple_the_best/article/details/77005400](https://blog.csdn.net/simple_the_best/article/details/77005400)
