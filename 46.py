#importing required libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import re
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# ===============================================
# |                Loading data                 |
# ===============================================
# Description: Loads train,test and valid data
# ===============================================

# read emoticon dataset
train_emoticon_df = pd.read_csv(r"datasets\train\train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
# read text sequence dataset
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str'].tolist()
train_seq_Y = train_seq_df['label'].tolist()

test_seq_df = pd.read_csv(r"datasets\test\test_text_seq.csv")
test_seq_X = test_seq_df['input_str'].tolist()

valid_seq_df = pd.read_csv(r"datasets\valid\valid_text_seq.csv")
# read feature dataset
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']

def print_description(description): print("\n" + f" {'=' * 5} {description} {'=' * 5} ".center(60, '=') + "\n")
print_description("Dataset information")

print(f"Train dataset size: ")
print(f"train_emoticon_X: {len(train_emoticon_X)} train_emoticon_Y: {len(train_emoticon_Y)}")
print(f"train_seq_X: {len(train_seq_X)} train_seq_Y: {len(train_seq_Y)}")
print(f"train_feat_X: {train_feat_X.shape} train_feat_Y: {train_feat_Y.shape}")
print()
print("Test dataset size: ")
print(f"test_emoticon_X: {len(test_emoticon_X)}")
print(f"test_seq_X: {len(test_seq_X)} ")
print(f"test_feat_X: {test_feat_X.shape}")

# ==============================================================================================
# |                        Emoticons as Features Dataset                                       |
# ==============================================================================================
# Description: Complete implementation of Emoticon-Features-Dataset Model                      |
# ==============================================================================================

# Loading the datasets
training_data = pd.read_csv(r'datasets\train\train_emoticon.csv')
testing_data = pd.read_csv(r'datasets\test\test_emoticon.csv')
validation_data=pd.read_csv(r'datasets\valid\valid_emoticon.csv')

#Taking output of training data and validation data in series objects named output_training_data && output_validation_data
output_training_data = training_data['label']
output_validation_data=validation_data['label']

#storing less important emojis in dictionary along with their counts
emojis_less_important = {
    '😛': 1, '🛐': 1, '🙼': 1, '😑': 2, '😣': 2, '🚼': 1, '🙯': 2
}


#function to remove less important emojis from the input string
def remove_less_important_emojis(input_string):
  """It takes a string as input and iterate over each emoji in dictionary based on
    its count and remove those emojis from input string and outputs the remaning string"""
  emojis_in_string = list(input_string)
  for emoji, count in emojis_less_important.items():
    for i in range(count):
      if emoji in emojis_in_string:
        emojis_in_string.remove(emoji)
  return ''.join(emojis_in_string)


# removing less important emojis from every row of training , validation and testing data and storing it in new column 'final_emoticon'
training_data['final_emoticon'] = training_data['input_emoticon'].apply(remove_less_important_emojis)
testing_data['final_emoticon'] = testing_data['input_emoticon'].apply(remove_less_important_emojis)
validation_data['final_emoticon']=validation_data['input_emoticon'].apply(remove_less_important_emojis)

# taking all unique emojis in data to the one set named unique emojis (after removal of less important emojis)
unique_emojis = set(''.join(training_data['final_emoticon'].values))

#storing this emojis in list in a sorted order
unique_emojis_in_list = sorted(unique_emojis)

#giving an each emoji one number and storing it in dictionary named emoji_indices
emoji_indices = {emoji: index for index, emoji in enumerate(unique_emojis_in_list)}

# Function to encode the data(emojis) with respect to position
def positional_encoding(in_string, maximum_length, emoji_indices):
    """"
    Created a 3D matrix with name positional_encoding_matrix which contains n 2D matrices
    where
          n is no of strings(no of emoji strings) which is passed in in_string.

    Each 2D matrix with shape (maximum_length , no of unique emojis)
     where
            maximum length is 3(length of string after removal of less important emojis)
            no of unique emojis is count of unique emojis identified in the training data

    Every time it takes one string from the in_string (which contain all strings)
    Starts iterating over that string

    If particular emojis is found in avialable emojis
    then mark the position of that emoji in coressponding 2D matrix's row as 1
    else skip this step

    return postional_encoding_matrix

    Using this both emoji and its position is encoded.
    """
    positional_encoding_matrix = np.zeros((len(in_string), maximum_length, len(emoji_indices)))
    for i,j in enumerate(in_string):
        for k, emoji in enumerate(j):
            if emoji not in emoji_indices:
                continue
            if k < maximum_length:  # Ensure we stay within the limit
                positional_encoding_matrix[i, k, emoji_indices[emoji]] = 1  # One-hot encoding
    return positional_encoding_matrix


# Get maximum length of the row after removal of less important emojis
maximum_length = training_data['final_emoticon'].apply(len).max()

# tranform the data to positional encoding format
training_x = positional_encoding(training_data['final_emoticon'], maximum_length, emoji_indices)
testing_x = positional_encoding(testing_data['final_emoticon'],maximum_length, emoji_indices)
validation_x=positional_encoding(validation_data['final_emoticon'],maximum_length, emoji_indices)

# Making the data flattend
"""training_x , testing_x and validation_x contains 3D matrices , this reshape converts each 2D matrix to one row"""
training_x_flattend = training_x.reshape(training_x.shape[0], -1)
testing_x_flattend = testing_x.reshape(testing_x.shape[0], -1)
validation_x_flattend = validation_x.reshape(validation_x.shape[0], -1)

# scaling the features
scaler = StandardScaler()
training_x_scaled = scaler.fit_transform(training_x_flattend)
testing_x_scaled = scaler.transform(testing_x_flattend)
validation_x_scaled = scaler.transform(validation_x_flattend)

# training the logistic regression model
logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(training_x_scaled, output_training_data)


#prediction on validation data
validation_output_predictions=logistic_regression_model.predict(validation_x_scaled)
print_description("Emoticon Dataset model classification report/results")
print(classification_report(output_validation_data,validation_output_predictions))

# predicting the testing data
testing_output_predictions = logistic_regression_model.predict(testing_x_scaled)
np.savetxt('pred_emoticon.txt', testing_output_predictions, fmt='%d')
print_description("Emoticon Dataset model Testing results are saved in pred_emoticon.txt file")

# ==============================================================================================
# |                        Deep Features Dataset                                               |
# ==============================================================================================
# Description: Complete implementation of Deep-Features-Dataset  Model                         |
# ==============================================================================================

# read feature train dataset
train_feat = np.load(r"datasets\train\train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

# read feature validation dataset
valid_feat = np.load(r"datasets\valid\valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

# read feature validation dataset
test_feat = np.load(r"datasets\test\test_feature.npz", allow_pickle=True)
test_feat_X = test_feat['features']

# flatten the features
train_feat_X_flat = train_feat_X.reshape(train_feat_X.shape[0], -1)
valid_feat_X_flat = valid_feat_X.reshape(valid_feat_X.shape[0], -1)
test_feat_X_flat = test_feat_X.reshape(test_feat_X.shape[0], -1)


def give_imp_rows(data,rows=[1,7,12]):
  """

  Function extract specific rows (default: rows 1, 7, and 12) and flatten it for each input feature.

  Parameters:
  data : numpy array
      A 3D array where each entry is 2D matrix.
  rows : list (optional)
      List of row indices to extract from each matrix. Defaults to [1, 7, 12].

  Returns:
  numpy array
      A 2D array where each entry contains the flatten array with specified rows.

  """
  imp_feature_rows=[]
  for i in range(data.shape[0]):
    # Extract the specified 'rows' from 2D matrix and flatten into 1D
    imp_feature_rows.append(data[i,rows].flatten())
  return np.array(imp_feature_rows)

train_feat_X_flat_imp_rows = give_imp_rows(train_feat_X)
valid_feat_X_flat_imp_rows = give_imp_rows(valid_feat_X)

# instantiate the logistic regression model without regulization
logreg_model = LogisticRegression(random_state=16, penalty=None)
# fit the model with data
logreg_model.fit(train_feat_X_flat_imp_rows, train_feat_Y)

print_description("Deep features Dataset model results")
# find accuracy on train data and validation data
print(f" Train accuracy : {logreg_model.score(train_feat_X_flat_imp_rows, train_feat_Y)} , Validation accuracy : {logreg_model.score(valid_feat_X_flat_imp_rows, valid_feat_Y)}")

# Extract important rows for the test data
test_feat_X_flat_imp_rows = give_imp_rows(test_feat_X)
# Make predictions on the test data
test_predictions = logreg_model.predict(test_feat_X_flat_imp_rows)
# Save the predictions to a text file
np.savetxt('pred_deepfeat.txt', test_predictions, fmt='%d')
print_description("Deep features Dataset model Testing results are saved in pred_deepfeat.txt file")

# ==============================================================================================
# |                        Text Sequence Dataset                                               |
# ==============================================================================================
# Description: Complete implementation of Text-Sequence-Dataset  Model                         |
# ==============================================================================================


def extract_tokens(seq_df):
    '''In this function finding the tokens
     from the input sequences given by finding common and
     frequently occurred substrings'''

    seq_strings = seq_df['input_str'].tolist()
    no_of_seq_strings = len(seq_strings)

    # freq_of_each_substring is used to store frequency of each substring

    freq_of_each_substring = {}

    # finding all substrings possible

    for string in seq_strings:

        # creating a set for substrings  so no duplication of substring that is already existed in list

        set_of_substrings = set()
        str_len = len(string)
        for i in range(str_len):
            for j in range(i + 1, str_len + 1):
                sub_string = string[i:j]
                set_of_substrings.add(sub_string)

        # finding frequency of each substring

        for sub_string in set_of_substrings:
            freq_of_each_substring[sub_string] = freq_of_each_substring.get(sub_string, 0) + 1

    # to find substrings that occur in all sequences

    substrings_in_all_seq = []
    for sub_string, frequency in freq_of_each_substring.items():
        if frequency == no_of_seq_strings:
            substrings_in_all_seq.append(sub_string)
    substrings_in_all_seq.sort(key=len, reverse=True)

    '''finding substrings by removing the smaller substrings
     which are part of already longer substrings'''

    substrings_unique_in_all_seq = []
    for sub_string in substrings_in_all_seq:
        unique_flag = True
        for unique_sub_string in substrings_unique_in_all_seq:
            if sub_string in unique_sub_string:
                unique_flag = False
        if unique_flag:
            substrings_unique_in_all_seq.append(sub_string)

    # removing the substring 000 if it is in all sequences

    if '000' in substrings_unique_in_all_seq:
        substrings_unique_in_all_seq.remove('000')

    # final_strings to store frequency of final strings to be considered

    final_strings = {}

    # created a regular expression from unique substrings

    pattern = '|'.join(map(re.escape, substrings_unique_in_all_seq))

    # applied regex pattern to input data, splits the data if any pattern matches

    seq_df['input_str_split'] = seq_df['input_str'].apply(lambda row: re.split(pattern, row))

    # finding substrings which are length less than are equal to 6

    for seq in seq_df['input_str_split']:
        for string in seq:
            if string and len(string) <= 6:
                final_strings[string] = final_strings.get(string, 0) + 1
    tokens_found = []

    # keeping as a token if frequency of substring is atleast five in entire data

    for string, freq in final_strings.items():
        if freq >= 5:
            tokens_found.append(string)

    tokens_found.extend(substrings_unique_in_all_seq)

    return tokens_found[3:]


def seq_Tokenizer(seq_df, tokens):
    '''In this function we apply tokenization on input string using the tokens found in extract_tokens
    function At end return the tokenized sequences in data frame with new column tokenized_seq'''

    tokens = set(tokens)

    '''In tokenize_each_seq function we match input strings with the extracted tokens
    and return tokens in that seq'''

    def tokenize_each_seq(seq):

        # removing leading zeros from sequence

        seq = seq.lstrip('0')

        # storing the tokens in the given sequence

        tokens_in_seq = []
        curr_index = 0
        seq_len = len(seq)

        '''In this while loop we are matching the input sequences with the tokens
        in order of finding longest tokens first to smallest tokens possible at last'''

        while curr_index < seq_len:
            token_found_flag = False

            # sort the tokens based on length of token

            for token in sorted(tokens, key=len, reverse=True):

                # checking if any token matches in current substring, if matched appending it to tokens_in_seq

                if seq.startswith(token, curr_index):
                    tokens_in_seq.append(token)
                    curr_index += len(token)
                    token_found_flag = True
                    break

            # if no token matched append [UNK]

            if not token_found_flag:
                tokens_in_seq.append("[UNK]")
                curr_index += 1
        return tokens_in_seq

    # applying the tokenize_each_seq to all sequences in data frame
    seq_df['tokens'] = seq_df['input_str'].apply(tokenize_each_seq)
    seq_df['tokenized_seq'] = seq_df['tokens'].apply(lambda token: ' '.join(token))

    return seq_df['tokenized_seq']


# Extracting tokens

tokens_extracted = extract_tokens(train_seq_df)

# applying the tokenization on training data

train_seq_df['tokenized_seq'] = seq_Tokenizer(train_seq_df, tokens_extracted)

# preparing training data

X_train = train_seq_df['tokenized_seq']
y_train = train_seq_df['label']

# Tokenizer on training data to convert text in to numerical sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)

# finding max length after tokenization and padded the sequences to have uniform length for all tokenized sequences

max_length = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')

# Preparing the validation data

valid_seq_df['tokenized_seq'] = seq_Tokenizer(valid_seq_df, tokens_extracted)
X_valid = valid_seq_df['tokenized_seq']
y_valid = valid_seq_df['label']
X_valid_seq = tokenizer.texts_to_sequences(X_valid)
X_valid_pad = pad_sequences(X_valid_seq, maxlen=max_length, padding='post')

# Preparing the testing data

test_seq_df['tokenized_seq'] = seq_Tokenizer(test_seq_df, tokens_extracted)
X_test = test_seq_df['tokenized_seq']
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

input_dim = len(tokenizer.word_index) + 1


def text_seq_model(input_dim, X_train, y_train, X_valid, y_valid, X_test_pad):
    # creating a neural network of sequential

    model = Sequential(name="text_seq_model")

    '''adding an embedding layer and here we are generating each token
     embedding vector length set to 8 '''

    model.add(Embedding(input_dim=input_dim, output_dim=8))

    '''flatten the Embeddings because before passing to dense layer 
    we should convert 2D output from embedding layer to 1D vector'''

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    '''early stopping used if there is no improvement in consecutive 15 
    epochs since patience is 15 we stop the model training'''

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_valid_pad, y_valid),
              callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_valid, y_valid)
    print_description("Text sequence Dataset model Validation results and model summary")
    print(f"Validation Accuracy: {accuracy:.2f}")
    model.summary()

    # Predicting on test and saving in to text file

    test_predictions = model.predict(X_test_pad)
    test_predictions = (test_predictions > 0.5).astype(int)
    np.savetxt('pred_textseq.txt', test_predictions, fmt='%d')
    print_description("Text sequence Dataset model Testing results are saved in pred_textseq.txt file")


print_description("Text sequence Dataset model Training")

# Model training and prediction test data
text_seq_model(input_dim, X_train_pad, y_train, X_valid_pad, y_valid, X_test_pad)


# ==============================================================================================
# |                                        Task-2                                              |
# ==============================================================================================
# Description: implementation of Combined model                                                |
# ==============================================================================================

# Combining the features from TASK-1 from all three models (concatenating along the last axis)

X_train_combined = np.hstack((X_train_pad, train_feat_X_flat, training_x_scaled))
X_valid_combined = np.hstack((X_valid_pad, valid_feat_X_flat, validation_x_scaled))
X_test_combined = np.hstack((X_test_pad, test_feat_X_flat, testing_x_scaled))

#target labels
y_train_combined = y_train
y_valid_combined = y_valid

#Loading pre-trained RandomForest model
with open(r"random_forest_model_0.9939.pickle", 'rb') as file:
    model = pickle.load(file)

#Make predictions on the validation set
y_valid_pred = model.predict(X_valid_combined)
print_description("Combined model Validation results")

# Evaluating the model
accuracy = accuracy_score(y_valid_combined, y_valid_pred)
print(f"Validation Accuracy with Combined Features: {accuracy:.4f}")

#Make predictions on the testing set
test_predictions = model.predict(X_test_combined)
np.savetxt('pred_combined.txt', test_predictions, fmt='%d')

print_description("Combined model Testing results are saved in pred_combined.txt file")













