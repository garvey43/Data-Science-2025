# -*- coding: utf-8 -*-
"""Assigmnent 11 .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BLBU5NJU4DVsoDPrvtZ7pMHZYHxNOVoW
"""

def remove_and_sort(Lin, k):
  del Lin[:k]
  Lin.sort()


L = [1,6,3,5,4,8]
k = 1

remove_and_sort(L, k)

print(L)