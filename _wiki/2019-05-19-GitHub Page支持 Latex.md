---
layout: post
title: GitHub Page 支持 Latex
categories: [GitHub, Latex]
description: GitHub Page 支持 Latex
keywords: Github, Latex, Page
---

GitHub Page 支持 Latex

修改 `_includes/header.html` 文件，在`<head></head>` 找一个地方添加以下内容：  
```
<!-- Latex -->
    <script type="text/x-mathjax-config"> 
      MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); 
    </script>
   <script type="text/x-mathjax-config">
     MathJax.Hub.Config({tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            processEscapes: true
          }
        });
   </script>
   
   <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript">
   </script>
```