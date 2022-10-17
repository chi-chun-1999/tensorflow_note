from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

print(sequences)

# %%  

one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')

print(one_hot_results.shape)
print(one_hot_results[0][:15])
print(one_hot_results[1][:15])


# %%  
word_index = tokenizer.word_index
print('Found %s unique tokens.'%len(word_index))
