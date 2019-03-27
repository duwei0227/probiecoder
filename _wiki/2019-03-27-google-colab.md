---
layout: post
title: Google Colaboratory
categories: [Colaboratory]
description: Google Colaboratory
keywords: Google, Drive, colab, Colaboratory
---

Google Colaboratory

### 1、colab挂载google drive
```python
from google.colab import drive
drive.mount('/content/gdrive')

# 切换到个人收藏目录
import os
os.chdir('/content/gdrive/My Drive')
```