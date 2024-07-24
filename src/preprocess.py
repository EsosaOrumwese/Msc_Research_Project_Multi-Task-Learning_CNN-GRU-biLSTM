import pandas as pd
import numpy as np
import os
import csv
import itertools
from scipy.interpolate import interp1d
from tqdm import tqdm
import torch


## basic functions for preprocessing
def segment_data(data):
      '''Gets the different journey segments based on the data. It doesn't do so
      per day. When assigning the segment number after identifying the segments, it
      doesn't take date into account'''

      data['Date'] = data['Date'].astype(str)
      data['Label_Change'] = (data.groupby(['Date', 'Position'], group_keys=False)['Coarse_label']   # group_keys=False prevents the result from being Multiindex
                              .apply(lambda x: x.shift(1) != x))
      data['Segment'] = data.groupby('Position')['Label_Change'].cumsum()

      return data

def create_sub_segments(data, window_size=112, overlap=0.5):
      '''Creates sub_segments for each segment (journey) using a specific window_size
      with overlap'''
      step_size = int(window_size * (1 - overlap))
      sub_segments = []
      for (user, position), user_position_data in data.groupby(['User', 'Position']):
            sub_segment_id = 1
            for segment, segment_data in user_position_data.groupby('Segment'):
                  for start in range(0, len(segment_data) - window_size + 1, step_size):
                        end = start + window_size
                        sub_segment = segment_data.iloc[start:end].copy()
                        sub_segment['Sub_Segment'] = sub_segment_id
                        sub_segments.append(sub_segment)
                        sub_segment_id += 1
      return pd.concat(sub_segments).reset_index(drop=True)


class LSTM_featureExtractor:
      '''Feature extractor for biLSTM model'''
      def __init__(self, subbed_data, num_features=6, window=112):
            self.subbed_data = subbed_data
            self.num_features = num_features
            self.window = window
            self.num_samples = len(subbed_data) // window
            self.lstm_features = np.empty((self.num_samples, window, num_features))
            self.lstm_labels = np.empty(self.num_samples)

      def feature_extractor(self, start=0):
            '''Extracts 112 sequence for each 6 features (inputs) for the biLSTM model.
            There is no explicit overlap applied here as the subbed_data already has
            50% overlap is implicity applied on it.'''


            for i in range(self.num_samples):
                  end = start + self.window
                  chunk = self.subbed_data.iloc[start:end, :]

                  # Extract features
                  self.lstm_features[i, :, 0] = chunk['acceleration_x'].values    # acceleration_x
                  self.lstm_features[i, :, 1] = chunk['acceleration_y'].values    # acceleration_y
                  self.lstm_features[i, :, 2] = chunk['acceleration_z'].values    # acceleration_z
                  self.lstm_features[i, :, 3] = chunk['gyroscope_x'].values       # gyroscope_x
                  self.lstm_features[i, :, 4] = chunk['gyroscope_y'].values       # gyroscope_y
                  self.lstm_features[i, :, 5] = chunk['gyroscope_z'].values       # gyroscope_z

                  # Extract label (assuming one label per chunk)
                  lab_list = np.unique(chunk['Coarse_label'].values)
                  if len(lab_list) == 1:
                        self.lstm_labels[i] = lab_list[0]
                  else:
                        # Handle case where there are multiple labels in a chunk
                        self.lstm_labels[i] = lab_list[0]  # or some other strategy to resolve conflicts

                  start = end

            return self.lstm_features, self.lstm_labels
      

class Long_Tranv_Angvel_from_Quarternions:
      '''Gets the longutinal acceleration and transverse acceleration from quaternions. Angular
      velocity is gotten from the gyroscope measurements'''

      def __init__(self) -> None:
            pass

      def euler_from_quaternions(self, q):
            '''
            Gets the Euler angles (roll, pitch and yaw) from quaternions using (ZYX notation)
            Input:
            q: Array which contains orientation coordinates w, x, y, z which corresponds to (q0, q1, q2, q3)
            Output:
            roll, pitch, and yaw in degrees
            '''
            q0, q1, q2, q3 = q

            # Roll (x-axis rotation)
            roll = np.arctan2(2 * (q1 * q0 + q2 * q3), q0**2 - q1**2 - q2**2 + q3**2)

            # Pitch (y-axis rotation)
            pitch = -np.arcsin(2 * (q1 * q3 - q2 * q0))

            # Yaw (z-axis rotation)
            yaw = np.arctan2(2 * (q3 * q0 + q1 * q2), q0**2 + q1**2 - q2**2 - q3**2)

            return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
      
      def get_rot_from_euler(self, sub_segment):
            '''Derives the rotation matrix given the euler angles'''
            roll_pitch_yaw_df = sub_segment.apply(self.euler_from_quaternions, axis=1, result_type='expand')
            roll, pitch, yaw = roll_pitch_yaw_df.median()
            
            # rotation matrix
            R = np.array([
                        [(np.cos(pitch)* np.cos(yaw)), (np.cos(pitch) * np.sin(yaw)), -np.sin(pitch)],
                        [((np.sin(roll)*np.sin(pitch)*np.cos(yaw)) - (np.cos(roll)*np.sin(yaw))), ((np.sin(roll)*np.sin(pitch)*np.sin(yaw)) + (np.cos(roll) * np.cos(yaw))), (np.sin(roll) * np.cos(pitch))],
                        [((np.cos(roll)*np.sin(pitch)*np.cos(yaw)) + (np.sin(roll)*np.sin(yaw))), ((np.cos(roll)*np.sin(pitch)*np.sin(yaw)) - (np.sin(roll) * np.cos(yaw))), (np.cos(roll) * np.cos(pitch))]
                        ])

            return R
            
      def quaternion_to_rotation_matrix_zyx(self, q):
            '''
            Get rotation matrix (ZYX notation) from quaternion
            Input:
            q: Array which contains orientation coordinates w, x, y, z which corresponds to (q0, q1, q2, q3)
            Output:
            rotation matrix (ZYX notation), numpy array
            '''
            q0, q1, q2, q3 = q

            # rotation matrix
            R = np.array([
                        [(q0**2 + q1**2 - q2**2 - q3**2), 2*(q0*q3 + q1*q2), 2*(q1*q3 - q0*q2)],
                        [2*(q1*q2 - q3*q0), (q0**2 - q1**2 + q2**2 - q3**2), 2*(q0*q1 + q2*q3)],
                        [2*(q0*q2 + q1*q3), 2*(q2*q3 - q0*q1), (q0**2 - q1**2 - q2**2 + q3**2)]
                        ])

            return R

      def get_rotation_matrix_sub_segment(self, sub_segment):
            '''
            For each sub_segment or sub_journey, we estimate the median quaternion and use it to create our rotation matrix
            Input:
                  sub_segment: Array containing orientation coordinates w, x, y, z for sub_segment. Expected shape
                              is (len(sub_segment), 4)
            Output:
                  rotation matrix (ZYX notation). numpy array
            '''
            q = sub_segment.median()

            R = self.quaternion_to_rotation_matrix_zyx(q)

            return R
      
      def rotate_acc_to_global_coord(self, data, window=None):
            '''
            Takes the dataset and for each sub_segment, it get's the rotation matrix (using the orientation sensor data), 
            to rotate the accelerometer triaxal data to the global coordinates. Now works with all coarse_labels
            Input:
                  data: dataframe with appropriate labels
                  window: the number of datapoints in each sub-segment
            Output:
                  dataframe with appended columns (longitudinal acceleration, transversal acceleration corresponding to a'_x, a'_y)
            '''
            if window is None:
                  window = data.groupby(['User', 'Position', 'Sub_Segment']).count()['Coarse_label'].unique()[0]
            
            ### The idea here is to halve the window (128) to 64 so as to adhere to info above;
            window = int(window/2)

            # get idx of columns in df
            cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                  'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z']
            col_idx = [data.columns.get_loc(col) for col in cols]

            start = 0
            for i in range(0, int(len(data)/window), 1):
                  end = start + window
                  chunk = data.iloc[start:end, col_idx].copy() # gives me the needed columns

                  # get rotation matrix and rotate acc vals to global coord
                  R = self.get_rot_from_euler(chunk.iloc[:,3:]) # sends in only the orientation data and results in rotation matrix
                  long_acc, tranv_acc = (chunk.iloc[:,:3] @ R)[0], (chunk.iloc[:,:3] @ R)[1]

                  # attach values to df
                  repl_idx_long = data.columns.get_loc('long_acc')
                  repl_idx_tranv = data.columns.get_loc('tranv_acc')
                  data.iloc[start:end, repl_idx_long] = long_acc
                  data.iloc[start:end, repl_idx_tranv] = tranv_acc

                  del chunk, long_acc, tranv_acc
                  start = end

            return data

      def get_angular_vel_from_gyr(self, data):
            '''
            Takes the dataset and get's the angular velocity which corresponds to the square root of the sum of the
            squared gyroscope measurements. Edited from working with just coarse_lavel==5.0 to all
            Input:
                  dataframe with appropriate labels
            Output:
                  dataframe with appended columns (ang_vel)
            '''
            # get idx of columns in df
            cols = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
            #col_idx = [data.columns.get_loc(col) for col in cols]
            
            # simply the sqrt of the sum of the squared gyro measurements
            ang_vel = np.sqrt(np.sum(data[cols]**2, axis=1))
            data['ang_vel'] = ang_vel

            return data
      
      def get_long_tranv_angl(self, data):
            '''Gets the longitudinal acc, tranversal acc and angular velocity for entire dataset'''
            #idx = data.query('Coarse_label == 5.0').index

            modified_data = self.get_angular_vel_from_gyr(self.rotate_acc_to_global_coord(data))#.loc[idx, :]))
            data = modified_data

            return data
      
      def add_transf_cols(self, data):
            '''Add empty columns to data'''
            data['long_acc'] = 0.0#np.nan
            data['tranv_acc'] = 0.0#np.nan
            data['ang_vel'] = 0.0#np.nan

            return data
      

class Prep_data_for_CNN:
      '''Prepare data for training with CNN'''
      def __init__(self) -> None:
            pass

      def combine_sequences(self, data):
            '''Combine sub_segments into *segments for each journey. Where journey is defined as a trip for 
            each user at each segment for each position.

            *segments here doesn't represent the original segment since the data was already
            windowed down with overlap previously.
            '''
            combined_sequences = {}
            users = data['User'].unique()
            positions = data['Position'].unique()

            # iterate through each combination of user and positions
            for user, position in itertools.product(users, positions):
                  coarse_labels = data.query("User == @user & Position==@position")['Coarse_label'].unique()
                  for coarse_label in coarse_labels:
                        # subset data
                        user_data = data.query("User == @user & Position==@position & Coarse_label==@coarse_label")
                        segments = user_data['Segment'].unique()
                        # extract all required data for each segment
                        for segment in segments:
                              segment_data = user_data.query("Segment == @segment")
                              long_acc = segment_data['long_acc'].values
                              tranv_acc = segment_data['tranv_acc'].values
                              ang_vel = segment_data['ang_vel'].values
                              
                              combined_sequences[(user, position, coarse_label, segment)] = {   
                                    'long_acc': long_acc,
                                    'tranv_acc': tranv_acc,
                                    'ang_vel': ang_vel
                                    }
            
            return combined_sequences

      def interpolate_to_length(self, array, target_length):
            '''Interpolate arrays that are smaller than the specificied window_size'''
            x = np.linspace(0, len(array) - 1, num=len(array))
            f = interp1d(x, array, kind='linear')
            x_new = np.linspace(0, len(array) - 1, num=target_length)
            interpolated_array = f(x_new)
            return interpolated_array

      def create_windows(self, data, window_size, overlap):
            '''Given window_size and overlap, it creates windows of data'''
            stride = int(window_size * (1 - overlap))
            num_windows = (len(data) - window_size) // stride + 1
            windows = np.array([data[i*stride:i*stride+window_size] for i in range(num_windows)])
            
            if len(windows) == 0:
                  # occurs in cases when num_windows == 0 or -1 because len(data) <= window_size
                  return np.array([self.interpolate_to_length(data, window_size)])
            
            # Check if the last window covers the end of the array by index
            last_window_end_index = (num_windows - 1) * stride + window_size - 1
            if last_window_end_index < len(data) - 1:
                  last_start_index = len(data) - window_size
                  last_window = data[last_start_index:]
                  if len(last_window) < window_size:
                        last_window = self.interpolate_to_length(last_window, window_size)
                  windows = list(windows)
                  windows.append(last_window)
            
            return np.array(windows)
      
      def get_interpolated_sections(self, subbed_data, windowed_df):
            '''Iterates over the data to see which of the sections have been interpolated'''
            users = windowed_df.user.unique()
            positions = windowed_df.position.unique()
            sections_interpltd = []

            for user, position in itertools.product(users, positions):
                  segments = subbed_data.query('User==@user & Position==@position').Segment.unique()
                  for segment in segments:
                        a = len(subbed_data.query('User==@user & Segment==@segment & Position==@position'))
                        b = len(windowed_df.query("user==@user & segment==@segment & position==@position"))

                        if a/224 != b:
                              sections_interpltd.append((user, position, segment))
            
            return sections_interpltd

      
      def get_windowed_df(self, data, window_size=224, overlap=0.25):
            '''Creates a dataframe which contains each *segment sequentially divided into windows of a specific length'''
            windowed_data = []
            combined_sequences = self.combine_sequences(data)
            
            #c=0
            for (user, position, coarse_label, segment), signals in combined_sequences.items():
                  long_windows = self.create_windows(signals['long_acc'], window_size, overlap)
                  tranv_windows = self.create_windows(signals['tranv_acc'], window_size, overlap)
                  ang_vel_windows = self.create_windows(signals['ang_vel'], window_size, overlap)

                  for i in range(long_windows.shape[0]):
                        windowed_data.append({
                              'user': user,
                              'coarse_label':coarse_label,
                              'segment': segment,
                              'position': position,
                              'window_index': i,#+c,
                              'long_acc_window': long_windows[i],
                              'tranv_acc_window': tranv_windows[i],
                              'ang_vel_window': ang_vel_windows[i]
                        })


            return pd.DataFrame(windowed_data)
      
      def prep_input_for_CNN(self, windowed_df):
            '''Prepares the windowed_df for training and testing with CNN. Returns
            X, y (encoded) and unique values'''
            X = []
            y = []

            for index, row in windowed_df.iterrows():
                  long_acc_window = row['long_acc_window']
                  tranv_acc_window = row['tranv_acc_window']
                  ang_vel_window = row['ang_vel_window']
                  combined_window = np.stack([long_acc_window, tranv_acc_window, ang_vel_window], axis=0)
                  X.append(combined_window)
                  y.append(row['user'])

            X = np.array(X)
            y_lab = np.array(y)
            uniq, y = np.unique(y_lab, return_inverse=True)
            return X, y, uniq
      
      def prep_for_FMAPextract(self, subbed_data, windowed_df):
            '''Prepares the windowed_df for feature maps extractions matching the amount of data to reflect the 
            number of data pts in the lstm features.'''
            # get sections in windowed_df which are interpolated. These wouldn't be duplicated
            sections_interpltd = self.get_interpolated_sections(subbed_data, windowed_df)
            windowed_df['Interp_Flag'] = 0

            for (user, position, segment) in sections_interpltd:
                  # notice above that segments for each position are unique so we don't really need coarse_label
                  # since full_windowed_df doesn't have one
                  idx = windowed_df.query("user==@user & position==@position & segment==@segment").index
                  windowed_df.loc[idx[-1], 'Interp_Flag'] = 1

            # Filter out rows where Interp_Flag == 1
            to_duplicate = windowed_df[windowed_df['Interp_Flag'] != 1].copy()

            # Concatenate the original DataFrame with the duplicated rows
            windowed_df = pd.concat([windowed_df, to_duplicate]).sort_index(kind='mergesort').reset_index(drop=True)  

            # adding duplicate flag to df
            windowed_df['Dupl_Flag'] = 0
            flag = [0, 1] * (len(windowed_df.query('Interp_Flag == 0'))//2)
            idx = windowed_df.query('Interp_Flag == 0').index

            windowed_df.loc[idx, 'Dupl_Flag'] = flag

            # add train_flag where interp==1 or interp==0 and dupl_flag=0
            windowed_df['Train_Flag'] = 0
            idx = windowed_df.query('(Interp_Flag == 1) | (Interp_Flag == 0 & Dupl_Flag==0)').index

            windowed_df.loc[idx, 'Train_Flag'] = 1

            users = windowed_df.user.unique()
            positions = windowed_df.position.unique()
            sections = []

            for user, position in itertools.product(users, positions):
                  segments = windowed_df.query('user==@user & position==@position').segment.unique()
                  for segment in segments:
                        sections.append((user, position, segment))

            ## turn off the flags all other sections
            idx_keep = []
            for (user, position, segment) in sections:
                  idx_keep.extend(windowed_df.query('user==@user & position==@position & segment==@segment').index)

            # select the rows that aren't driving data
            idx_flip = windowed_df.index[~windowed_df.index.isin(idx_keep)]

            # flip them to 0 so that they don't train
            windowed_df.loc[idx_flip, 'Train_Flag'] = 0

            return windowed_df

      ## Function for getting window_df from data_dir
      def get_window_df_from_dir(self, data_dir, window_size=224, overlap=0, prep_for_FMAPextract=False):
            '''Given data directory for the sub-segmented dataset, it returns the windowed_df
            used to derive data for training the SimpleCNN 1d-2D mapper'''

            # read data
            subbed_data = pd.read_csv(data_dir)

            # add long_acc, tranv_acc and ang_vel to columns
            subbed_data = Long_Tranv_Angvel_from_Quarternions().add_transf_cols(subbed_data)
            subbed_data = Long_Tranv_Angvel_from_Quarternions().get_long_tranv_angl(subbed_data)

            windowed_df = self.get_windowed_df(subbed_data, window_size, overlap)

            if prep_for_FMAPextract == True:
                  windowed_df = self.prep_for_FMAPextract(subbed_data, windowed_df)

            # change other labels to `Not_driving`  (a dummy label)
            windowed_df['user_act'] = windowed_df['user'].values
            idx_users = windowed_df.query('coarse_label == 5.0').index
            df_with_drivers = windowed_df.index.isin(idx_users)
            windowed_df.loc[~df_with_drivers, 'user'] = 'Not_driving'

            return windowed_df
      

class FeatureMaps_extractor:
      def __init__(self, base_dir, device, idx_list, split):
            '''Class for extracting 3-channel images from each corresponding model for each 1-D set of signals'''
            self.base_dir = base_dir
            self.device = device
            self.idx_list = idx_list

            # Create directories and CSV files if they don't exist
            splits = ['train', 'valid', 'test']
            for split in splits:
                  split_dir = os.path.join(self.base_dir, split)
                  os.makedirs(split_dir, exist_ok=True)
                  csv_file = os.path.join(split_dir, 'metadata.csv')
                  if not os.path.exists(csv_file):
                        with open(csv_file, mode='w', newline='') as file:
                              writer = csv.writer(file)
                              writer.writerow(['filename', 'label'])


      def get_feature_map(self, model, x_batch):
            '''Extracts feature map from a given model'''
            # Place model in evaluation mode
            model.eval()
            
            # Hook to store the feature map
            feature_maps = []

            # Define a hook function to capture the feature map
            def hook(module, input, output):
                  feature_maps.append(output.detach().cpu().numpy())

            # Register the hook to the desired layer
            hook_handle = model.conv2.register_forward_hook(hook)

            try:
                  # Run the data through the model to trigger the hook
                  with torch.no_grad():
                        model(x_batch)
            finally:
                  # Remove the hook after feature extraction
                  hook_handle.remove()

            return feature_maps[0]
      
      def feature_map_extractor(self, models, dataloaders, split):
            '''Extracts feature maps and saves them as .npy files, updating a CSV with the filenames and labels.
            It receives a dictionary of models and a dictionary of dataloaders. It assumes batch_size=1'''
            
            idx = 0
            # Get iterators for each dataloader
            iterators = {key: iter(dataloaders[key]) for key in dataloaders.keys()}

            # iterate through the assumed number of batches the dataloader
            for _ in tqdm(range(len(self.idx_list)), desc=f"Extracting Feature Maps {split}"):
                  feature_maps = {}
                  labels = {}

                  # Extract feature maps and labels for each modality
                  for key in dataloaders.keys():
                        x_batch, y_batch = next(iterators[key])
                        x_batch = x_batch.to(self.device)

                        f_maps = self.get_feature_map(models[key], x_batch)
                        feature_maps[key] = f_maps.squeeze(0).squeeze(1)  # Shape: (224, 224)
                        labels[key] = y_batch.numpy()

                  # Save the combined feature map for each sample in the batch
                  for i in range(x_batch.shape[0]):
                        combined_feature_maps = {key: feature_maps[key][i] for key in feature_maps}
                        self.save_combined_feature_map(combined_feature_maps, labels['long'][i], split, self.idx_list[idx])
                        idx += 1

      
      def save_combined_feature_map(self, feature_maps, label, split, idx):
            """
            Combines feature maps from different modalities into a single 3-channel input tensor,
            saves it as a .npy file, and logs the file name and label in a CSV file.

            Args:
            - feature_maps (dict): Dictionary containing feature maps for 'long', 'tranv', and 'angvel'.
            - label (int): Label for the current sample.
            - split (str): Data split ('train', 'valid', or 'test').
            - idx (int): Index for file naming. Gotten from Dataframe index
            """
            # Extract feature maps
            long_maps = feature_maps['long']
            tranv_maps = feature_maps['tranv']
            angvel_maps = feature_maps['angvel']

            # Combine the feature maps along the channel dimension
            combined_map = np.stack((long_maps, tranv_maps, angvel_maps), axis=0)  # Shape: (3, 224, 224)

            # Save the combined feature map
            filename = f'{idx}.npy'
            save_path = os.path.join(self.base_dir, split, filename)
            np.save(save_path, combined_map)

            # Log the file name and label in the CSV
            csv_file = os.path.join(self.base_dir, split, 'metadata.csv')
            with open(csv_file, mode='a', newline='') as file:
                  writer = csv.writer(file)
                  writer.writerow([filename, label])