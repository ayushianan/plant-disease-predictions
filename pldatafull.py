#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *


# In[3]:


path = Path('data/plantdatafull')
#path = untar_data(URLs.PETS,dest=dest)
path


# In[4]:


path.ls()


# In[5]:


path_train = Path('data/plantdatafull/train')
path_valid = Path('data/plantdatafull/valid')


# In[ ]:


#ImageDataBunch.from_folder??


# In[6]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[7]:


data.show_batch(rows=3, figsize=(15,6))


# In[8]:


data.classes


# In[9]:


data.c


# In[10]:


len(data.train_ds), len(data.valid_ds)


# In[11]:


for folder in data.classes:
    verify_images(path_train/folder, delete=True)


# In[12]:


for folder in data.classes:
    verify_images(path_valid/folder, delete=True)


# In[ ]:


#?cnn_learner


# In[13]:


learn = cnn_learner(data, models.resnet34, metrics = error_rate)


# In[14]:


learn.model


# In[15]:


learn.fit_one_cycle(3)


# In[16]:


learn.save('stage-1')


# In[17]:


interp = ClassificationInterpretation.from_learner(learn)

losses, indices = interp.top_losses()


# In[19]:


interp.plot_top_losses(9, figsize=(35,11), heatmap = False)


# In[ ]:


doc(interp.plot_top_losses)


# In[21]:


interp.plot_confusion_matrix(figsize=(45,20))


# In[24]:


interp.most_confused(2)


# In[25]:


learn.unfreeze()
learn.fit_one_cycle(1)


# In[26]:


learn.save('stage-2')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


learn.lr_find()


# In[28]:


learn.recorder.plot()


# In[29]:


learn.unfreeze()
learn.fit_one_cycle(1, max_lr=slice(3e-6,5e-5))


# In[30]:


learn.save('stage-3')


# In[31]:


interp = ClassificationInterpretation.from_learner(learn)

losses, indices = interp.top_losses()


# In[32]:


interp.plot_top_losses(9, figsize=(35,11), heatmap = False)


# In[35]:


interp.most_confused()


# In[36]:


len(data.train_ds), len(data.valid_ds)


# In[ ]:




