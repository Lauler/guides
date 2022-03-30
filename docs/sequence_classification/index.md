--- 
title: "Document Classification with Transformers"
author: "Faton Rekathati, KBLab"
date: "2022-03-30"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, references.bib]
biblio-style: apalike
link-citations: yes
description: "This is a guide describing how to use KBLab's transformer language models for document classification."
github-repo: lauler/guides
cover-image: ../images/kb_logo_text_black.png
url: 'https\://github.com/lauler/guides/sentiment_analysis/'
---



# Welcome {-}

KBLab is a creator of open source AI models and tools. We publish and share models 
that we have trained under the organizations [KB](https://huggingface.co/KB) and [KBLab](https://huggingface.co/KBLab) on [huggingface.co](https://huggingface.co). 

While the *pre-training* of these models require substantial compute resources and data, 
using an already pre-trained model and customizing it for a specific task is considerably less resource intensive. 
Most tasks will not require more than a regular consumer workstation to produce impressive results. 

At KBLab we would like for more people to be able to make practical use our models -- both in research and in an applied setting. 
Thus in an effort to make our models more accessible, we are publishing a series of guides demonstrating their usage by way of example on real world datasets. 
In this guide we cover **document classification**: the task of assigning a document to either one of two categories (**binary classification**) 
or one of several available categories (**multinomial classification**). 

Throughout this guide you will encounter colorful boxes providing advice and tips for the reader. 
Below, we briefly explain the purpose and function of each box. 

:::info
This box contains notes and extras with useful commentary on the concepts we cover. 
It is not strictly necessary to read the information in the green boxes to follow along with the examples, but it might help with your understanding. 
:::

:::help
We use the orange boxes to describe and link to useful external resources where you can learn more. 
They are also used to present alternative code solutions.
:::

:::warning
The red boxes alert the reader about common mistakes and pitfalls, in the hopes that the reader can
avoid having to suffer through some of the same glorious debugging sessions the authors had to endure.
:::


<script type="text/javascript">
title=document.getElementById('header');
title.innerHTML = '<img src="../images/kb_logo_text_black.png" alt="Test Image" style="max-width: 130px;">' + title.innerHTML
</script>
