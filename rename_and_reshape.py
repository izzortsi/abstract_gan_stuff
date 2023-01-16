#%%


from PIL import Image
import os

data_path = os.path.join(os.getcwd(), './data')
new_path = os.path.join(os.getcwd(), './data_new')
if not os.path.exists(new_path):
    os.makedirs(new_path)
    
# %%

for i, file in enumerate(os.listdir(data_path)):
    img = Image.open(os.path.join(data_path, file))
    new_image = img.resize((512, 512))
    new_image.save(os.path.join(new_path, f'{i:04d}.jpg'))
# %%
