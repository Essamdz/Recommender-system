import json
import pandas as pd
import re
import numpy as np
import random
import datetime
import pickle
import os
import glob
import time
import urllib
import socket

import json 
import os

from PIL import Image
import matplotlib.pyplot as plt


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse



max_out=150                 # Maximum number of  recommended images for each user
 
readit=False
text_out=False              #True  for GPU and text excluding 
sim_users_percentage=0.05   # The lowest similarity between users 

path="images/"
url_commenter="commenter0.json"
url_sharer="sharer0.json"
url_poster="poster0.json"
url_liker="liker0.json"



def dt(s):
    ls=s.split('T')
    d=ls[0].split('-')
    x=ls[1].split('+')
    t=x[0].split(':')
    return int(d[0][2:]+d[1]+d[2]+t[0]+t[1]+t[2])
    
def to_dt(x):
    ls=re.findall('\d\d',str(x))
    return('20'+ls[0]+"-"+ls[1]+"-"+ls[2]+"  "+ls[3]+":"+ls[4]+":"+ls[5])

def construct(url,post,user):
    ls=[]
    with open(url, encoding="utf-8") as fp:
       line = fp.readline()
       cnt = 1
       ls.append(line.strip())
       while line:
           line = fp.readline()
           cnt += 1
           if line:
               ls.append(line.strip())
    #print(len(ls))
    ls3=[]
    f=0
    for dic in ls: 
        tm=json.loads(dic)
        try:
            if  tm[post]['type']=='photo':
                im=tm[post]['id'] #['$oid']
                us=tm[user]['id']
                dat=dt(tm[post]['created_at'] )#['$date']['$numberLong']
                url=tm[post]["photo"]["path"]
                ls3.append((us,im,dat,url))
        except:
            f=f+1
    #print(f)
    return ls3


def download_images(new,path):
    print()
    print("Downloading files...............")
    socket.setdefaulttimeout(20) 
    ls4=list(new)
    fail=[]
    j=0
    gf=0
    for i in range(len(ls4)):
        try:
            ex=ls4[i][1].split(".")[-1]
            if len(ex)>5:ex=".jpg"
            urllib.request.urlretrieve(ls4[i][1], path+ls4[i][0]+"."+ex)#".png")
            if ex=="gif":
                Image.open(path+ls4[i][0]+"."+ex).convert('RGB').save(path+ls4[i][0]+".jpg")
                gf=gf+1
                #fail.append((ls4[i][0],ls4[i][1]))
                os.remove(path+ls4[i][0]+"."+ex)
            j=j+1
            if j%500==0:print(j, "files have been downloaded")
        except:
            print("downloading photo no",i," fails ")
            fail.append((ls4[i][0],ls4[i][1]))
    print(j," files are downloaded successfully")
    print(gf, " gif files are converted to jpg")
    return fail

def delete_images(delet): 
    del_list=list(delet)
    j=0
    for i in range(len(del_list)):
        fileList = glob.glob(path+del_list[i]+"*")
    
        for filePath in fileList:
            try:
                os.remove(filePath)
                j=j+1
            except:
                print("Error while deleting file : ", filePath)
    print(j," files are deleted successfully")


def reset(path):
    j=0
    fileList = glob.glob(path+"*.*")
    for filePath in fileList:
        try:
            os.remove(filePath)
            j=j+1
        except:
            print("Error while deleting file : ", filePath)
    print(j," files are deleted successfully")
    last_url_set=set()
    with open('set_url', 'wb') as f:
        pickle.dump(last_url_set, f)
    
    with_text=set()    
    with open('set_text', 'wb') as f:
        pickle.dump(with_text, f)
        

###### Date of each Image Dictionary {im: date} ########################
def image_date(commenter,sharer,create,liker):
    im_dic={}
    for x in commenter:
        im_dic[x[1]]=x[2]
    for x in sharer:
        im_dic[x[1]]=x[2]
    for x in create:
        im_dic[x[1]]=x[2]
    for x in liker:
        im_dic[x[1]]=x[2]
    return im_dic
    
############################# Generate The engagement matrix ##################   
def create_matrix(users,images,user_dict,images_dict,commenter,sharer,liker,create):    
    matrix = np.zeros([len(users), len(images)], dtype = float)    
    
    for x in commenter:
        matrix[user_dict[x[0]]][images_dict[x[1]]]=5
    
    
    for x in sharer:
        if matrix[user_dict[x[0]]][images_dict[x[1]]]==0:
            matrix[user_dict[x[0]]][images_dict[x[1]]]=5
    
    for x in liker:
        if matrix[user_dict[x[0]]][images_dict[x[1]]]==0:
            matrix[user_dict[x[0]]][images_dict[x[1]]]=4
        
    for x in create:
        if matrix[user_dict[x[0]]][images_dict[x[1]]]==0:
            matrix[user_dict[x[0]]][images_dict[x[1]]]=1
    
    matrix2=matrix.copy()
    
    q=matrix.sum(axis=1)       
    for i in range(len(users)):
        matrix2[i][:]=(matrix[i][:]/q[i]*max_out)
    return matrix2

 
############  Dictioray { user : engagement set  } ###########################

def user_engagement(matrix2):
    user_engage={i:[] for i in range(matrix2.shape[0])}
    for i in range(matrix2.shape[0]):
        for j in range(matrix2.shape[1]):
            if matrix2[i][j]!=0:
                user_engage[i].append(j)
    return user_engage
    


################### Order a dictionary by date {user ind: sorted results by date} ##############################
       
def order_by_date(dic_list, im_dic, images_dict_r):
    
    #################### date associate with images as tuple of list ##########
    im_cluster_tuple={}
    for i in range(len(dic_list)):
        lis_tuple=[]
        for j in dic_list[i]:
            lis_tuple.append((j,im_dic[images_dict_r[j]]))
        im_cluster_tuple[i]=lis_tuple
    ################# Soring the list of tuples according to date #############

    im_cluster_tuple_sorted={}
    for i in range(len(im_cluster_tuple)):
        ls_tp=im_cluster_tuple[i]
        u=sorted(ls_tp, key=lambda x: x[1],reverse=True)
        im_cluster_tuple_sorted[i]=u 
        
    ############## Dictionary { user ind : list of sorted results according to date } ###################
    
    im_cluster_sorted={}
    for i in range(len(dic_list)):
        lis2=[]
        for x in im_cluster_tuple_sorted[i]:
           lis2.append(x[0])
        im_cluster_sorted[i]=lis2
    return im_cluster_sorted

 


################ to Image ID  ###################################################
######## Dictioray { user : list of recommendation as id} ###############################
def to_id(user_res_index,users,images_dict_r):
    user_res_id={i:[] for i in range(len(users))}
    for i in range(len(user_res_index))  :
        u=[images_dict_r[y] for y in user_res_index[i]]
        user_res_id[i]=u
    return user_res_id



############################### Save Json ###############################
def save_Json(user_res_id):
    user_res_id2={user_dict_r[x]:user_res_id[x]  for x in user_res_id.keys()}
    with open('data_collaborative.json', 'w') as f:
        json.dump(user_res_id2, f, indent=4)
#######################################################################################



       
commenter=construct(url_commenter, post='Post', user='Commenter' )
sharer=construct(url_sharer, post='Original Post', user='Sharer' )
create=construct(url_poster, post='Post', user='Post Owner' )
liker=construct(url_liker, post='Post', user='Liker' )

all=set(commenter)|set(sharer)|set(create)|set(liker)

new_url_set={(x[1],x[3])  for x in all}


############################################################################
    
commenter=[(x[0],x[1],x[2]) for x in commenter ]#if x[1] not in (fail_set |with_text)]
commenter_set={x[0] for x in commenter}
commenter_set_im={x[1] for x in commenter}

sharer=[(x[0],x[1],x[2]) for x in sharer]# if x[1] not in (fail_set |with_text)]
sharer_set={x[0] for x in sharer}
sharer_set_im={x[1] for x in sharer}

create=[(x[0],x[1],x[2]) for x in create]# if x[1] not in (fail_set |with_text)]
create_set={x[0] for x in create}
create_set_im={x[1] for x in create}

liker=[(x[0],x[1],x[2]) for x in liker]# if x[1] not in (fail_set |with_text)]
liker_set={x[0] for x in liker}
liker_set_im={x[1] for x in liker}



############### All Users and All Images #######################
users=list(commenter_set|sharer_set|create_set|liker_set)
print("\nNumber of all users is ",len(users))
images=list(commenter_set_im|sharer_set_im|create_set_im|liker_set_im)
print("Number of all images is ",len(images))



################ indexing users and images dictionary ####################
user_dict = {item:ind for ind,item in enumerate(users)}
user_dict_r = {ind:item for ind,item in enumerate(users)}

images_dict = {item:ind for ind,item in enumerate(images)}
images_dict_r = {ind:item for ind,item in enumerate(images)}


im_dic=image_date(commenter,sharer,create,liker) 
   
matrix2= create_matrix(users,images,user_dict,images_dict,commenter,sharer,liker,create)                   



user_engage=user_engagement(matrix2)



#######################################################################


matrix2[matrix2>1]=1
data_items = pd.DataFrame(matrix2.transpose())

#------------------------
# User-User CALCULATIONS
#------------------------

# As a first step we normalize the user vectors to unit vectors.

# magnitude = sqrt(x2 + y2 + z2 + ...)
magnitude = np.sqrt(np.square(data_items).sum(axis=1))

# unitvector = (x / magnitude, y / magnitude, z / magnitude, ...)
data_items = data_items.divide(magnitude, axis='index')

def calculate_similarity(data_items):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim

# Build the similarity matrix
data_matrix = calculate_similarity(data_items)



def engage_recommendation(user_id,th3=max_out,th2=sim_users_percentage):
    
    
    id=user_dict[user_id]
    #user items
    arr_us=matrix2[id]
    us_item={i for i in range(len(arr_us)) if arr_us[i]!=0}
    
    
    
    user_sim=data_matrix.loc[id].nlargest(100)#len(users))
    sim=list(user_sim)
    us=list(user_sim.index)
    
    st_all=set()
    for i in range(len(us)):
        if sim[i]<th2:break
        arr_us=matrix2[us[i]]
        st_item={i for i in range(len(arr_us)) if arr_us[i]!=0}
        st_all_old=st_all
        st_all=st_all | st_item
        if len(st_all)-len(us_item)>th3:break
    st_all=st_all-us_item
    if len(st_all)>th3:st_all=st_all_old-us_item
    
    if not os.path.exists(user_id):
        os.mkdir(user_id)
        os.mkdir(user_id+'/recommendation')
        os.mkdir(user_id+'/engagement') 

    
    img_id_list=[ images_dict_r[x]   for x in us_item]
    url_eng={(x,dict(new_url_set)[x]) for x in img_id_list }
    download_images(url_eng,user_id+'/engagement/')
    

    
    img_id_list=[ images_dict_r[x]   for x in st_all]
    url_recom={(x,dict(new_url_set)[x]) for x in img_id_list }
    download_images(url_recom,user_id+'/recommendation/')



######## Dictioray { user as index : list of recommendation as index} ###############################

def  recommend_index(th3=max_out,th2=sim_users_percentage):
    user_recomm_list={}
    for ind in range(len(users)):
        
        if ind%1000==0:print(ind)

        arr_us=matrix2[ind]
        us_item={i for i in range(len(arr_us)) if arr_us[i]!=0}
    
        user_sim=data_matrix.loc[ind].nlargest(100)#len(users))
        sim=list(user_sim)
        us=list(user_sim.index)
        
        st_all=set()
        for i in range(len(us)):
            if sim[i]<th2:break
            arr_us=matrix2[us[i]]
            st_item={i for i in range(len(arr_us)) if arr_us[i]!=0}
            st_all_old=st_all
            st_all=st_all | st_item

            if len(st_all)-len(us_item)>th3:break
        st_all=st_all-us_item
        if len(st_all)>th3:st_all=st_all_old-us_item
        
        user_recomm_list[ind]=st_all
 
        
    return user_recomm_list



user_recomm_list=recommend_index()
user_recomm_list=order_by_date(user_recomm_list, im_dic, images_dict_r)  
user_res_id=to_id(user_recomm_list,users,images_dict_r)    
save_Json(user_res_id)


xx=len([1 for x in user_res_id.values() if len(x)>0])/len(user_res_id)
print("The ratio of users with recommendation :", xx)


#engage_recommendation(user_dict_r[1],100)
#engage_recommendation('5de6a8f70d290d12aa6db6b2',100)
#engage_recommendation('5fd13e110d290d2f496a4d06',100,0)


