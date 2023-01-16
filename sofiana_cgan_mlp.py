# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
from os import mkdir, makedirs, getcwd, listdir
from os.path import join, exists
from typing import overload
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import datetime

#%%


LOAD_MODEL = True
GPU = False
DEBUG = False
DATA_FORMAT = "xlsx"

## TRAINING AND MODEL PARAMETERS
batch_size = 100
epochs = 1000
gen_batch_size = 2000
# z_dim = NUM_FEATS*NUM_ME_FEATS
# z_dim = 20
# feat_dim = (lambda X: X.shape[1])(X)
z_dim = 60

loss_function = tf.keras.losses.BinaryCrossentropy(from_logits = True)

feature_scaler = StandardScaler()
me_feature_scaler = StandardScaler()

#%%
# DATA_PATH = "new_loans_history.xlsx"
DATA_PATH = f"20000_selected_data.{DATA_FORMAT}"
FUTURE_ME_DATA_PATH = "future_macroeconomic_scenarios.xlsx"
models_dir = 'models'
current_model_checkpoint_dir = os.path.join(models_dir, \
                                                f'mlp_cgan')
if not os.path.exists(models_dir):
  os.makedirs(models_dir)
LOANS_DATA_PATH = join(getcwd(), DATA_PATH)
MEV_PROJECTION_PATH = join(getcwd(), FUTURE_ME_DATA_PATH)

# FEATURES = []
FEATURES = ["original_interest_rate", "interest_rate", "original_balance", "balance", "loan_amount", "payment_amount"]
#m_real_gdp_growth m_real_disp_income_growth m_unemployment_rate m_cpi_inflation_rate m_treasury_rate_3_months m_mortgage_rate m_dow_jones	m_cre_price_index m_market_volatility_index
# ME_FEATURES = ["m_real_gdp_growth", "m_real_disp_income_growth", "m_unemployment_rate", "m_cpi_inflation_rate", "m_treasury_rate_3_months", "m_mortgage_rate", "m_dow_jones", "m_cre_price_index", "m_market_volatility_index"]
ME_FEATURES = ["m_unemployment_rate", "m_cpi_inflation_rate", "m_treasury_rate_3_months", "m_mortgage_rate", "m_dow_jones", "m_market_volatility_index"]
NUM_FEATS = len(FEATURES)
NUM_ME_FEATS = len(ME_FEATURES)
## Other parameters
READ_NUM_ROWS = 40000

#%%
if DATA_FORMAT == "xlsx":
  DATA = pd.read_excel(LOANS_DATA_PATH)
# DATA = pd.read_excel(LOANS_DATA_PATH, nrows=READ_NUM_ROWS)
elif DATA_FORMAT == "csv":
  DATA = pd.read_csv(LOANS_DATA_PATH)

#%%
if GPU:
  device_name = tf.config.list_physical_devices("GPU")[0].name
  tf.device(device_name)

if DEBUG:

  tf.debugging.set_log_device_placement(True)

  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)
  
  print(c)
  
  tf.debugging.set_log_device_placement(False)  

if len(FEATURES) > 0:
    FEATURES_DATA = DATA[[column_label for column_label in FEATURES if (column_label[:2] != "m_" and column_label != "the_date")]].copy()
else:
    FEATURES_DATA = DATA[[column_label for column_label in DATA.columns if (column_label[:2] != "m_" and column_label != "the_date")]].copy()
if len(ME_FEATURES) > 0:
    ME_FEATURES_DATA = DATA[ME_FEATURES].copy()
else:
    ME_FEATURES_DATA = DATA[[column_label for column_label in DATA.columns if column_label[:2] == "m_"]].copy()    

DATA[["the_date"]] = DATA[["the_date"]].astype(str)
DATE_DATA = DATA[["the_date"]].copy()
DATE_DATA = DATE_DATA.astype(str)
# %%

###DATA PREPROCESSING
features_data = FEATURES_DATA.values.astype(np.float32).copy()
me_features_data = ME_FEATURES_DATA.values.astype(np.float32).copy()
# date_data = np.datetime_as_string(DATE_DATA.values.copy())
date_data = DATE_DATA.values.copy()
# NUM_FEATS = features_data.shape[1]
# NUM_ME_FEATS = me_features_data.shape[1]


features_data = feature_scaler.fit_transform(features_data)
me_features_data = me_feature_scaler.fit_transform(me_features_data)

# X = np.hstack((features_data, me_features_data))
X = features_data
Y = me_features_data
X.shape
feat_dim = X.shape[1]
me_feat_dim = Y.shape[1]

# x_train, x_test = train_test_split(X, test_size=0.1, random_state=42)
x_train = X.copy()
y_train = Y.copy()
# Noise for visualization
z_vis = tf.random.normal([10, z_dim]) #o que acontece se eu mudar o 10?
x_iter = iter(tf.data.Dataset.from_tensor_slices((x_train, y_train, date_data)).shuffle(4 * batch_size).batch(batch_size).repeat())
#%%

def Generator():
  z = tf.keras.layers.Input(shape=(z_dim,), dtype='float32')
  y = tf.keras.layers.Input(shape=(NUM_ME_FEATS,), dtype='float32')
  d = tf.keras.layers.Input(shape=(1,), dtype='string')
  tr = tf.keras.layers.Input(shape=(1,), dtype='bool')

  x = tf.keras.layers.concatenate([z, y])
  x = tf.keras.layers.Dense(z_dim * NUM_ME_FEATS, activation="relu")(x)
  x = tf.keras.layers.Dense(NUM_FEATS, activation='sigmoid')(x)
  
  return tf.keras.Model(inputs=[z, y, d, tr], outputs=(x, y, d))


def Discriminator():
  X = tf.keras.layers.Input(shape=(NUM_FEATS,), dtype='float32')
  Y = tf.keras.layers.Input(shape=(NUM_ME_FEATS,), dtype='float32')
  tr = tf.keras.layers.Input(shape=(1,), dtype='bool')
  
  # y = tf.tile(tf.reshape(Y,[-1, 1, 1, NUM_DATES]), [1, 28, 28, 1])
  # y = Y
  # x = tf.keras.layers.Reshape((NUM_FEATS + NUM_ME_FEATS,))(X)
  x = tf.keras.layers.concatenate([X, Y])
  x = tf.keras.layers.Dense(NUM_FEATS * NUM_ME_FEATS, activation="relu")(x)
  out = tf.keras.layers.Dense(1)(x)
  return tf.keras.Model(inputs=[X, Y, tr], outputs=out)


G = Generator()
D = Discriminator()


# Loss functions

def G_loss(D, x_fake, y_real):
  return loss_function(tf.ones_like(D([x_fake, y_real, True])), D([x_fake, y_real, True]))
def D_loss(D, x_real, x_fake, y_real):
  return loss_function(tf.ones_like(D([x_real, y_real, True])), D([x_real, y_real, True])) + loss_function(tf.zeros_like(D([x_fake, y_real, True])), D([x_fake, y_real, True]))

# Optimizers
G_opt = tf.keras.optimizers.Adam(1e-4)
D_opt = tf.keras.optimizers.Adam(1e-4)
#%%

if not LOAD_MODEL:
# Train
  for epoch in range(epochs):
    z_mb = tf.random.normal([batch_size, z_dim])
    x_real, y_real, date = next(x_iter)
    # Record operations
    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:  
      x_fake, y, date = G([z_mb, y_real, date, True])
      G_loss_curr = G_loss(D, x_fake, y_real)
      D_loss_curr = D_loss(D, x_real, x_fake, y_real)
    # Gradients
    G_grad = G_tape.gradient(G_loss_curr, G.trainable_variables)
    D_grad = D_tape.gradient(D_loss_curr, D.trainable_variables)
    # Apply gradients
    G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
    D_opt.apply_gradients(zip(D_grad, D.trainable_variables))
    
    if epoch % 100 == 0:
      # Print results
      print('epoch: {}; G_loss: {:.6f}; D_loss: {:.6f}'.format(epoch+1, G_loss_curr, D_loss_curr))
      # print(np.sqrt(np.sum((x_train.mean(axis=0) - x_fake.numpy().mean(axis=0))**2)))
  G.save_weights(current_model_checkpoint_dir)    
  print("Model saved to {}".format(current_model_checkpoint_dir))
else:
  print("Model loaded from {}".format(current_model_checkpoint_dir))
  G.load_weights(current_model_checkpoint_dir)    
# %%
gen_x_iter = iter(tf.data.Dataset.from_tensor_slices((x_train, y_train, date_data)).shuffle(gen_batch_size).batch(gen_batch_size).repeat())

z_mb = tf.random.normal([gen_batch_size, z_dim])
x_real, y_real, date = next(gen_x_iter)
batch_generated, ys, date = G([z_mb, y_real, date, False])
assert np.alltrue(ys == y_real)

# %%
x_real = x_real.numpy()
y_real = y_real.numpy()
# %%

x_real = feature_scaler.inverse_transform(x_real)
y_real = me_feature_scaler.inverse_transform(y_real)
# %%
x_real

# %%
y_real
# %%

batch_generated = feature_scaler.inverse_transform(batch_generated.numpy())
ys = me_feature_scaler.inverse_transform(ys.numpy())

# %%


date.numpy().astype(np.datetime64)
#%%

df = pd.DataFrame(data = np.hstack((date, batch_generated,ys)), columns=["the_date"]+(list(FEATURES_DATA.columns)+list(ME_FEATURES_DATA.columns))
)
df_corr = pd.DataFrame(data = np.hstack((batch_generated,ys)), columns=(list(FEATURES_DATA.columns)+list(ME_FEATURES_DATA.columns))
)
df_corr_matrix = df_corr.corr()

# %%
df_corr_matrix
# %%
df.sort_values(by=["the_date"], inplace=True, ignore_index=True)
# %%
# df_ = df.drop(columns=["the_date"])


# df_.corr()


df_array = df.drop(columns=["the_date"]).to_numpy().astype(np.float32)


# %%

original_df = pd.DataFrame(data = np.hstack((date, x_real, y_real)), columns=["the_date"]+(list(FEATURES_DATA.columns)+list(ME_FEATURES_DATA.columns)))
original_df_corr = pd.DataFrame(data = np.hstack((x_real, y_real)), columns=(list(FEATURES_DATA.columns)+list(ME_FEATURES_DATA.columns)))
original_df.sort_values(by=["the_date"], inplace=True, ignore_index=True)

# %%
original_df_corr_matrix =original_df_corr.corr()
original_df_corr_matrix
#%%
np.abs(original_df_corr_matrix - df_corr_matrix).mean(axis=0).mean()


# %%
odf_array = original_df.drop(columns=["the_date"]).to_numpy().astype(np.float32)
#%%
odf_array


from scipy.special import rel_entr
# %%
generated_distribution = df_array
original_distribution =odf_array
#%%

rel_entr(original_distribution, generated_distribution).mean(axis=0)

# %%
rel_entr(original_distribution, generated_distribution).shape

# %%
from scipy.stats import entropy

entropy(original_distribution, generated_distribution)

# %%
# %%

# %%
df
# %%
np.alltrue(df.the_date == original_df.the_date)
# %%
# original_df.drop(columns=["the_date"], inplace=True)

# %%
df.tail()
# %%
original_df.tail()
# %%

# %%
original_df[original_df.the_date == b'2018-08-31']

# %%
DATA[DATA.the_date == '2018-08-31']

# %%
df.to_excel("generated_data.xlsx")

# %%
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(original_df_corr_matrix.round(1), annot=True)
plt.show()
# %%
sn.heatmap(df_corr_matrix.round(1), annot=True)
plt.show()

# %%
