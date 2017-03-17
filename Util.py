#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class contains useful functions 
# and fields.
#
# copyright Paul Baumann
#############################################

import Database_Handler
import numpy

TASK_METRIC_LABELS = ['NSP+Accuracy', 'NSP+Fscore', 'NSP+MCC', 'NST+Accuracy', 'NST+Fscore', 'NST+MCC', 'NP+Accuracy', 'NP+Fscore', 'NP+MCC']

DEMO_GROUP_LABELS = ['all', 'u_female','u_male','u_work','u_study','u_16_21','u_22_27',
                  'u_28_33','u_34_38','u_39_44','u_no_kids','u_par',
                  'u_par_f','u_par_m','u_single','u_family'];
                  
demo_groups = ['all', 'female','male','working','study','age_group_16_21','age_group_22_27',
                  'age_group_28_33','age_group_34_38','age_group_39_44','no_children_all','with_children_all',
                  'with_children_female','with_children_male','single','family'];

def areUsersBelongToDemoGroup(users, demo_group):
    
    mask_users_belong_to_demo_group = numpy.zeros((len(users),),dtype=bool)
    
    for idx in range(len(mask_users_belong_to_demo_group)):
        mask_users_belong_to_demo_group[idx] = userBelongToDemoGroup(users[idx], demo_group)
        
    return mask_users_belong_to_demo_group
    
    
def userBelongToDemoGroup(user, demo_group):

    ## get demographic data
    dbHandler = Database_Handler.Get_DB_Handler()
    query = ("select * FROM Demographics where userid = %s") % (user) 
    demographics = numpy.array(dbHandler.select(query))
    
    gender = demographics[0, 1]
    age = demographics[0, 2] 
    work = demographics[0, 3]
    relationship = demographics[0, 4]
    children = demographics[0, 5]
    
    if demo_group == 'all':
        return True
    if demo_group == 'female':
        return gender == 1
    if demo_group == 'male':
        return gender == 2
    if demo_group == 'working':
        return work == 1
    if demo_group == 'study':
        return work == 4
    if demo_group == 'age_group_16_21':
        return age == 2
    if demo_group == 'age_group_22_27':
        return age == 3
    if demo_group == 'age_group_28_33':
        return age == 4
    if demo_group == 'age_group_34_38':
        return age == 5
    if demo_group == 'age_group_39_44':
        return age == 6
    if demo_group == 'no_children_all':
        return children == 0
    if demo_group == 'with_children_all':
        return children > 0
    if demo_group == 'with_children_female':
        return (children > 0) & (gender == 1)
    if demo_group == 'with_children_male':
        return (children > 0) & (gender == 2)
    if demo_group == 'single':
        return relationship == 0
    if demo_group == 'family':
        return relationship > 0
    
def adjust_boxplot(bp):
    
    MY_LINEWIDTH = 5
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(linewidth=MY_LINEWIDTH)
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(linewidth=MY_LINEWIDTH)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(linewidth=MY_LINEWIDTH)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(linewidth=MY_LINEWIDTH)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', alpha=0.9, color = 'red')
        
        
    
