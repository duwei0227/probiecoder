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

#### 4、启动异常
如果启动时提示如下异常：
```
raceback (most recent call last):
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/traitlets/traitlets.py", line 528, in get
    value = obj._trait_values[self.name]
KeyError: 'allow_remote_access'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 864, in _default_allow#c.NotebookApp.file_to_run = ''
_remote
    addr = ipaddress.ip_address(self.ip)
  File "/home/duwei0227/anaconda3/lib/python3.7/ipaddress.py", line 54, in ip_address
    address)
ValueError: '' does not appear to be an IPv4 or IPv6 address

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/duwei0227/anaconda3/bin/jupyter-notebook", line 11, in <module>
    sys.exit(main())
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/jupyter_core/application.py", line 266, in launch_instance
    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/traitlets/config/application.py", line 657, in launch_instance
    app.initialize(argv)
  File "<decorator-gen-7>", line 2, in initialize
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/traitlets/config/application.py", line 87, in catch_config_error
    return method(app, *args, **kwargs)
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 1628, in initialize
    self.init_webapp()
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 1378, in init_webapp
    self.jinja_environment_options,
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 159, in __init__
    default_url, settings_overrides, jinja_env_options)
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 252, in init_settings
    allow_remote_access=jupyter_app.allow_remote_access,
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/traitlets/traitlets.py", line 556, in __get__
    return self.get(obj, cls)
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/traitlets/traitlets.py", line 535, in get
    value = self._validate(obj, dynamic_default())
  File "/home/duwei0227/anaconda3/lib/python3.7/site-packages/notebook/notebookapp.py", line 867, in _default_allow_remote
    for info in socket.getaddrinfo(self.ip, self.port, 0, socket.SOCK_STREAM):
  File "/home/duwei0227/anaconda3/lib/python3.7/socket.py", line 748, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno -2] Name or service not known
```
可以修改 `c.NotebookApp.ip = '0.0.0.0'`  
  
   
   
参考：  
[https://blog.csdn.net/simple_the_best/article/details/77005400](https://blog.csdn.net/simple_the_best/article/details/77005400)
[https://blog.csdn.net/u013042928/article/details/83382336](https://blog.csdn.net/u013042928/article/details/83382336)