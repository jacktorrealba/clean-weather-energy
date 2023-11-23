# import modules
# for general data analysis/plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# for data prep
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.stats import zscore

# neural net modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# for model evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

########################################################################################################

# defining the dataframe 
df = pd.read_csv('cleanweather_energy.csv')
df.info()

########################################################################################################

# SUBSETTING AND DATA FORMAT

# dropping the 'gust' column since it won't be used in the analysis
df.drop(['gust'], axis=1, inplace=True)

# rename columns
df = df.rename(columns={'tmpf' : 'AirTemp(F)', 'dwpf' : 'DewPoint(F)', 'relh' : 'RelHumidity', 'drct' : 'WindDirection', 'sknt' : 'WindSpeed', 'HE' : 'HourofDay', 'mslp' : 'SeaLevelPressure'})

# take a look at the distributions
# df.hist()
# plt.rcParams["figure.figsize"] = (15,15)
# plt.tight_layout()
# plt.show()
########################################################################################################

# FEATURE ENGINEERING

# address outliers in wind speed and wind direction
# get z-score of wind speed and wind direction
z_scores = zscore(df[['WindSpeed', 'WindDirection']])

# set a threshold
threshold = 3 

# identify the outliers above the threshold
outliers = (np.abs(z_scores) > threshold).any(axis=1)

# remove the outliers from the dataframe
df = df[~outliers]

# defining variables used for modeling
x = df[['AirTemp(F)', 'DewPoint(F)', 'RelHumidity', 'WindDirection', 'WindSpeed', 'HourofDay', 'SeaLevelPressure']]
y = df['MWh'] # target variable

# perform log transformation
x['log_SeaLevelPressure'] = np.log1p(x['SeaLevelPressure'])
x.drop(['SeaLevelPressure'] ,axis=1, inplace=True)

# apply box-cox transformation to WindDirection
x['boxcox_WindDirection'], _ = boxcox(x['WindDirection'] + 1)  # Adding 1 to avoid issues with zero values

# apply box-cox transformation to WindSpeed
x['boxcox_WindSpeed'], _ = boxcox(x['WindSpeed'] + 1) # Adding 1 to avoid issues with zero values

# drop old columns
x.drop(['WindSpeed'] ,axis=1, inplace=True)
x.drop(['WindDirection'] ,axis=1, inplace=True)
########################################################################################################

# VISUALIZATIONS - after feature engineering

# plotting distributions of x variables to check if distribution of variables may be skewed or contain outliers
x.hist()
plt.rcParams["figure.figsize"] = (15,15)
plt.tight_layout()
plt.show()

########################################################################################################

# MODEL PREPERATION

# converting to numpy array
x = np.array(x)
y = np.array(y)

# splitting the data (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=16118)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# using min max scaling
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)

########################################################################################################

# BUILDING THE MODEL
print(x_train.shape, y_train.shape)
# defining the input shape to avoid hard coding it
x_train.shape[1]

# building the model
model = Sequential()
model.add(Dense(64, input_shape = (x_train.shape[1],), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.summary()

# compiling the model
model.compile(optimizer='RMSprop', loss = 'mse', metrics = ['mae'])

# early stopping callback
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True)

# fitting the model
history = model.fit(x_train, y_train,
                    validation_data = (x_test, y_test),
                    callbacks=[es],
                    epochs = 800,
                    batch_size = 8,
                    shuffle = True,
                    verbose = 1)

# accuracy between training and validation
history_dict = history.history
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# let's see the training and validation accuracy by epoch
history_dict = history.history
loss_values = history_dict['mae'] 
val_loss_values = history_dict['val_mae'] 
epochs = range(1, len(loss_values) + 1) 
plt.plot(epochs, loss_values, 'bo', label='Training mae')
plt.plot(epochs, val_loss_values, 'orange', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.legend()
plt.show()

# scatterplot of actual vs. pred
fig, axes = plt.subplots(1,2)

# training partition results
axes[0].scatter(x=y_train, y=model.predict(x_train)) 
axes[0].set_xlabel("Actual", fontsize=10)
axes[0].set_ylabel("Predicted",  fontsize=10)
axes[0].set_title("Training")
# add 45 deg line
x = np.linspace(*axes[0].get_xlim())
axes[0].plot(x, x, color='purple')


# validation partition results
axes[1].scatter(x=y_test, y=model.predict(x_test)) 
axes[1].set_xlabel("Actual", fontsize=10)
axes[1].set_ylabel("Predicted",  fontsize=10)
axes[1].set_title("Validation")
# add 45 degree line
x = np.linspace(*axes[1].get_xlim())
axes[1].plot(x, x, color='red')

# tight layout
fig.tight_layout()

# show the plot
plt.show()

# storing model results
pred = model.predict(x_test)
trainpreds = model.predict(x_train)

# R2
print('Training R2:', r2_score(y_train, trainpreds))
print('Test R2:', r2_score(y_test, pred))

# MAE
print('Training MAE:', mean_absolute_error(y_train, trainpreds))
print('Test MAE:', mean_absolute_error(y_test, pred))

# RMSE
print('Training RMSE:' , mean_squared_error(y_train, trainpreds))
print('Test RMSE:' , mean_squared_error(y_test, pred))

# accuracy between training and validation
history_dict = history.history
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()