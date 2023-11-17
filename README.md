# Deep-Learing-Components
Phan Duc Hung 20214903

# Infer
Step 1: Join competition https://www.kaggle.com/competitions/bkai-igh-neopolyp to create notebook

Step 2: Download model and libraries

```python
!pip install torchsummary
!pip install torchgeometry
!pip install segmentation-models-pytorch
import requests
import os

url = 'https://drive.google.com/uc?id=1aIQaLP3OeJvaYpn-skLNW9ytJcqsfbID&export=download&confirm=t&uuid=4426acaa-01ff-49e2-8507-550e89a31654'

save_dir = '/kaggle/working/'

response = requests.get(url)

with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

```
Step 3: Infer
```python
!git clone https://github.com/hwnginsoict/Deep-Learing-Components
!cp /kaggle/working/model.pth /kaggle/working/Deep-Learning-Components
!python /kaggle/working/Deep-Learing-Components/infer.py
```
