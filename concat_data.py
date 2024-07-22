import pandas as pd
from src.preprocess import segment_data

## Combines the data from all users together and selected required columns

user1_file = './data/User_1_downsampled_dataset.csv'
user2_file = './data/User_2_downsampled_dataset.csv'
user3_file = './data/User_3_downsampled_dataset.csv'

users = ['User1', 'User2', 'User3']
columns = ['time_ms', 'acceleration_x', 'acceleration_y', 'acceleration_z',
           'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
           'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
           'Date', 'Position', 'Coarse_label', 'Fine_label',
           'Segment', 'User']

concat_df = pd.DataFrame(columns=columns)


for i, user_file in enumerate([user1_file, user2_file, user3_file]):
      data = pd.read_csv(user_file)

      # removing NaN coarse_label
      keep_rows = data[data['Coarse_label'] != 0.0].index
      data = data.iloc[keep_rows, :]
      data = data.reset_index(drop=True)

      # add user to data
      data['User'] = users[i]

      # add segments to the data to represent journeys
      data = segment_data(data)

      concat_df = pd.concat([concat_df, data], ignore_index=True)

      del data

# Save the combined dataset
concat_df[columns].to_csv('./data/concat_users.csv', index=False, header=True)

del concat_df


