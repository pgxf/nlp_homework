import nltk
import os
training_data_dir = './data/original_data'
words_data_dir = './data/processed_data'
filenames = [name for name in os.listdir(training_data_dir)
             if os.path.isfile(os.path.join(training_data_dir, name))]
corpus_list = []
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for name in filenames:
    with open(os.path.join(training_data_dir, name), 'rt', encoding='Latin-1') as f:
        try:
            data = f.read()
        except UnicodeDecodeError as e:
            print(name+' '+str(e))
        with open(os.path.join(words_data_dir, name), 'wt', encoding='Latin-1') as wf:  # 先分词再分句
            wf.write('\n'.join(sent_tokenizer.tokenize(
                ' '.join(nltk.word_tokenize(data)))))
