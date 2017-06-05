#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:12 2017

@author: nadim
"""

class message():
    def __init__(self,file_name):
        self.file_name = file_name


    def list_of_messages(self):
        with open(self.file_name) as f:
            lines = f.readlines()
  
        for i in range(0,len(lines)):
            lines[i]=lines[i].split(" ")
       
        
        list =[]
        for k in range(0,len(lines)):
            for j in range(0,len(lines[k])):
                list.append(lines[k][j])
    
        
        for i in range(0,len(list)):
            if len(list[i])<2 :
                list[i]= "bk" 
        return list

    # -*- coding: utf-8 -*-

