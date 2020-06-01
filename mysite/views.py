import base64
import os
import pickle

import torch
from transformers import BertModel, DistilBertModel
from kobert_transformers.tokenization_kobert import KoBertTokenizer
from transformers import *

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.generic import TemplateView
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
from tika import parser
import nltk
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
from konlpy.tag import Kkma
# from konlpy.utils import pprint
# from gensim.models import word2vec

# Create your views here.
from mysite.models import Words, Collocations, Embeddings

RESULT_SUCCESS = 1
RESULT_FAIL = -1
RESULT_ERROR = 0


class Home(TemplateView):
    template_name = "home.html"


def db_view(request):
    if request.method == 'GET':
        return render(request, 'db_view.html')
    if request.method == 'POST':
        if not request.POST:
            return render(request, 'db_view.html')

        unwanted_words = request.POST.getlist('checks[]')
        year = request.POST['year']
        top_count = int(request.POST['top_count'])
        text_type = request.POST['text_type']

        print("Unwanted words: {}".format(unwanted_words))
        if unwanted_words:
            set_unwanted_words(unwanted_words, text_type)

        if text_type == 'words':
            data = Words.objects.all().filter(year=year).order_by('-count').values('text', 'count', 'year')[:top_count]
        else:
            data = Collocations.objects.all().filter(year=year).order_by('-count').values('text', 'count', 'year')[:top_count]

        if data.__len__() == 0:
            return render(request, 'no_data.html')

        context = {}
        context['words'] = data
        context['result'] = RESULT_SUCCESS
        return render(request, 'db_view.html', context)


def set_unwanted_words(words, text_type):
    # adding unwanted word to stopwords_ko.txt and removing from DB
    with open("stopwords_ko.txt", 'a+', encoding="UTF-8") as f:
        for word in words:
            f.write(word + "\n")
            if text_type == 'words':
                Words.objects.filter(text=word).delete()
            else:
                Collocations.objects.filter(text=word).delete()


def visualize(request):
    if request.method == 'GET':
        # extract_similarities_for_each_text()
        return render(request, 'visualization.html')
    if request.method == 'POST':
        if not request.POST:
            return render(request, 'visualization.html')

        year = request.POST['year']
        top_count = int(request.POST['top_count'])
        text_type = request.POST['text_type']
        topic_num = int(request.POST['topic_num'])
        wc_dir = 'media/tmp_wordcloud'

        context = {}

        # region Prepare data for Chart
        year_start = int(request.POST['year_start'])
        year_end = int(request.POST['year_end'])
        top5_words_for_sel_year = Words.objects.filter(year=year).order_by('-count')[:5]
        top5_collocations_for_sel_year = Collocations.objects.filter(year=year).order_by('-count')[:5]

        if not top5_words_for_sel_year.__len__() == 0:
            # get data for words chart data
            res = get_words_chart_data(top5_words_for_sel_year, year_start, year_end)
            context['word_chart_data'] = res['word_chart_data']
            context['is_word_chart_data_exists'] = res['is_word_chart_data_exists']

        if not top5_collocations_for_sel_year.__len__() == 0:
            # get data for collocations chart data
            res = get_collocations_chart_data(top5_collocations_for_sel_year, year_start, year_end)
            context['collocation_chart_data'] = res['collocation_chart_data']
            context['is_collocation_chart_data_exists'] = res['is_collocation_chart_data_exists']
        # endregion

        # region Draw Word Cloud
        word_freq = {}
        word_cloud_count = 100
        wc_data = Words.objects.all().filter(year=year).order_by('-count').values('text', 'count', 'year')[:word_cloud_count]

        for item in wc_data:
            word_freq[item['text']] = item['count']

        mask = np.array(Image.open(os.path.join(wc_dir, "cloud.png")))
        stopwords = set(STOPWORDS)
        wc = WordCloud(mask=mask, font_path='C:\\Users\\NSL\\Downloads\\Nanum_Gothic\\NanumGothic-Regular.ttf', background_color="white",
                       max_words=word_cloud_count,
                       stopwords=stopwords)

        if not word_freq.__len__() == 0:
            wc.generate_from_frequencies(word_freq)
            wc.to_file(os.path.join(wc_dir, 'wc.png'))
            context['is_wc_data_exits'] = 1  # shows that data for word cloud exists

        # endregion

        # region Prepare data for topic
        if text_type == 'words':
            data = Words.objects.all().filter(year=year).order_by('-count')[:top_count]
        else:
            data = Collocations.objects.all().filter(year=year).order_by('-count')[:top_count]

        if not data.__len__() == 0:
            context['topics_data'] = kmeans_clustering(data, topic_num)['topics_data']
            context['is_topics_data_exits'] = 1
        # endregion

        context['result'] = RESULT_SUCCESS
        return render(request, 'visualization.html', context)


def get_words_chart_data(top5_words_for_sel_year, year_start, year_end):
    words = []
    result = {}
    result['word_chart_data'] = []
    for word in top5_words_for_sel_year:
        words.append(Words.objects.filter(text=word.text, year__range=[year_start, year_end]).order_by('year'))

    dif_years = False
    for w in words:
        counter_year = year_start
        counter_words = 0
        for i in range(0, (year_end - year_start) + 1):
            if counter_year > year_end:
                break

            if counter_words < len(w):
                if w[counter_words].year == counter_year:
                    # final_data.append({"text": w[counter_words].text, "year": w[counter_words].year, "count": w[counter_words].count})
                    txt = w[counter_words].text
                    yr = w[counter_words].year
                    cnt = w[counter_words].count

                    if not dif_years:
                        counter_words = i + 1
                    else:
                        counter_words = counter_words + 1
                    # dif_years = False
                else:
                    dif_years = True
                    # final_data.append({"text": w[counter_words].text, "year": counter_year, "count": 0})
                    txt = w[counter_words].text
                    yr = counter_year
                    cnt = 0
            else:
                # final_data.append({"text": w[counter_words].text, "year": counter_year, "count": 0})
                if counter_words >0:
                    txt = w[counter_words - 1].text
                yr = counter_year
                cnt = 0

            result['word_chart_data'].append({"text": txt, "year": yr, "count": cnt})
            counter_year += 1

    result['is_word_chart_data_exists'] = 1
    return result


def get_collocations_chart_data(top5_collocations_for_sel_year, year_start, year_end):
    result = {}
    collocations = []
    result['collocation_chart_data'] = []
    for collocation in top5_collocations_for_sel_year:
        collocations.append(Collocations.objects.filter(text=collocation.text, year__range=[year_start, year_end]).order_by('year'))

    dif_years = False
    for c in collocations:
        counter_year = year_start
        counter_collocations = 0
        for i in range(0, (year_end - year_start) + 1):
            if counter_year > year_end:
                break

            if counter_collocations < len(c):
                if c[counter_collocations].year == counter_year:
                    # final_data.append({"text": w[counter_words].text, "year": w[counter_words].year, "count": w[counter_words].count})
                    txt = c[counter_collocations].text
                    yr = c[counter_collocations].year
                    cnt = c[counter_collocations].count

                    if not dif_years:
                        counter_collocations = i + 1
                    else:
                        counter_collocations = counter_collocations + 1
                    # dif_years = False
                else:
                    dif_years = True
                    # final_data.append({"text": w[counter_words].text, "year": counter_year, "count": 0})
                    txt = c[counter_collocations].text
                    yr = counter_year
                    cnt = 0
            else:
                # final_data.append({"text": w[counter_words].text, "year": counter_year, "count": 0})
                txt = c[counter_collocations - 1].text
                yr = counter_year
                cnt = 0

            result['collocation_chart_data'].append({"text": txt, "year": yr, "count": cnt})
            counter_year += 1

    result['is_collocation_chart_data_exists'] = 1

    return result


def upload(request):
    context = {}
    if request.method == 'POST':
        if not request.FILES and not request.POST["txt_input"] and not request.POST["keywords_input"]:
            context['result'] = RESULT_ERROR
            return render(request, 'upload.html', context)

        year = int(request.POST['year'])

        # handling Files input
        if request.FILES:
            print("Processing File...")
            fs = FileSystemStorage()
            files_withno_text = []
            for uploaded_file in request.FILES.getlist('documents'):
                files = os.listdir('media')
                filename = uploaded_file.name
                print("Filename: " + filename)

                # remove space in filename if exists
                if " " in filename:
                    filename = uploaded_file.name.replace(" ", "")

                if filename in files:
                    context['result'] = RESULT_FAIL
                    context['filename'] = filename
                    return render(request, 'upload.html', context)

                name = fs.save(filename, uploaded_file)

                parsed = parser.from_file('./media/' + filename)  # read the file

                # check if the extracted content from file is string type. If not just skip this file
                if isinstance(parsed["content"], str):
                    process_text(parsed["content"], year)
                else:
                    files_withno_text.append(filename)
                    continue
            context['no_text_files'] = files_withno_text
            print("Files with no-string content: {}".format(files_withno_text))

        # handling Text input
        if request.POST["txt_input"]:
            print("Processing text input...")
            txt = request.POST["txt_input"]
            process_text(txt, year)

        # handling Keywords input
        if request.POST["keywords_input"]:
            print("Processing keywords input...")
            txt = request.POST["keywords_input"]
            process_keywords(txt, year)

        context['result'] = RESULT_SUCCESS
    return render(request, 'upload.html', context)


def process_keywords(text, year):
    txt_array = text.split(",")
    for item in txt_array:
        if len(item.split()) > 1:  # save as a collocation
            new_ngram, created = Collocations.objects.get_or_create(text=item, year=year)
            if created:
                new_ngram.count = 1
                new_ngram.save()
            else:
                new_ngram.count += 1
                new_ngram.save()

        else:  # save as a word
            new_word, created = Words.objects.get_or_create(text=item, year=year)
            if created:
                new_word.count = 1
                new_word.save()
            else:
                new_word.count += 1
                new_word.save()


def process_text(text, year):
    # reg_exp_date = re.compile(r'([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))')
    # year = 0
    # if reg_exp_date.search(filename):
    #     year = int(reg_exp_date.search(filename).group().split('.')[0])

    print("1. Saving words...")
    save_words(pre_process(text), year)

    print("2. Saving 2-grams...")
    save_n_grams(pre_process(text), year, 2)

    print("2. Saving 3-grams...")
    save_n_grams(pre_process(text), year, 3)

    # konlpy_module(parsed["content"], year)
    # word2vec_function(parsed["content"].replace('\n', ' '))


def save_words(corpus, year):
    words_array = re.findall('\\w+', corpus.replace('\n', ''))  # take all the words using regular expression

    for word_item in words_array[1:]:  # just get rid of the first word since it's some garbage value
        new_word, created = Words.objects.get_or_create(text=word_item, year=year)
        if created:
            new_word.count = 1
            new_word.save()
        else:
            new_word.count += 1
            new_word.save()

    # TODO: do following after all words and collocations are uploaded
    # print("1-2. Saving embeddings for words based on similarities...")
    # save_similarities(words_array)


def save_n_grams(corpus, year, n):
    tokens = nltk.word_tokenize(corpus)

    # make bigram by default
    n_gram_measures = nltk.collocations.BigramAssocMeasures()
    n_gram_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    # make trigrams if n=3
    if n == 3:
        n_gram_measures = nltk.collocations.TrigramAssocMeasures()
        n_gram_finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

    # apply filters to finder
    n_gram_finder.apply_freq_filter(2)

    ngrams = n_gram_finder.nbest(n_gram_measures.pmi, 50)  # find top 50 collocations from words

    collocations_data = []
    for ngram in ngrams:
        new_ngram, created = Collocations.objects.get_or_create(text=" ".join(map(str, ngram)), year=year)
        collocations_data.append(" ".join(map(str, ngram)))
        if created:
            new_ngram.count = 1
            new_ngram.save()
        else:
            new_ngram.count += 1
            new_ngram.save()

    # TODO: do following after all words and collocations are uploaded
    # print("2-2. Saving embeddings for n-grams based on similarities...")
    # save_similarities(collocations_data)


def save_similarities(tokens):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')  # tokenize using Bert tokenizer
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')  # load Bert pre-trained model

    for token in tokens:
        # extract IDs for each word/collocation using Bert tokenizer
        input_ids = torch.tensor(tokenizer.encode(token)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)  # take output from pre-trained Bert model
        last_hidden_states = outputs[0]  # take word/collocation embeddings (vectors) (shape: 1x768)
        np_arr = np.array(last_hidden_states[0].detach().numpy())  # convert to numpy array

        # take mean of all embeddings for the current word/collocation to make one representing vector
        np_arr = np.mean(np_arr, axis=0)

        # encode the result to bytes to save into database
        np_bytes = pickle.dumps(np_arr)
        np_base64 = base64.b64encode(np_bytes)
        new_item, created = Embeddings.objects.get_or_create(text=token, embedding=np_base64)
        if created:
            np_bytes = base64.b64decode(new_item.embedding)  # encoded array
            np_array = pickle.loads(np_bytes)  # actual decoded array


def kmeans_clustering(texts, num_of_topic):
    words = []
    years = []
    embeddings = []
    result = {}
    result['topics_data'] = []
    for text in texts:
        vector = Embeddings.objects.filter(text=text.text)
        if not vector.__len__() == 0:
            words.append(vector[0].text)
            years.append(text.year)
            np_bytes = base64.b64decode(vector[0].embedding)
            embeddings.append(pickle.loads(np_bytes))
        # else:
        # print("No data: " + text.text)

    clustering = KMeans(n_clusters=num_of_topic)
    if not embeddings.__len__() == 0:
        clustering.fit(embeddings)
        for i in range(0, num_of_topic):
            n_cluster_data = ClusterIndicesNumpy(clustNum=i, labels_array=clustering.labels_)
            # print("***** Cluster {} *******".format(i))
            for n in n_cluster_data:
                result['topics_data'].append({"topic": str(i + 1), "text": words[n], "year": years[n]})

    return result


def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


def pre_process(corpus):
    corpus = corpus.lower()
    stop_words_set = []
    stop_collocations_set = []
    with open("stopwords_ko.txt", 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.split(" ").__len__() > 1:
                stop_collocations_set.append(line[:-1])
            else:
                stop_words_set.append(line[:-1])

    corpus = " ".join([i for i in nltk.regexp_tokenize(corpus, '\\w+') if i not in stop_words_set])

    for col in stop_collocations_set:
        if col in corpus:
            corpus = re.sub(col + " ", '', corpus)
    return corpus


def plot_similarity():
    # region Setting font for matplotlib
    font_dirs = ['C:\\Users\\NSL\\Downloads\\Nanum_Gothic\\', ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)
    mpl.rcParams['font.family'] = 'NanumGothic'
    # endregion

    words = []
    embeddings = []
    similarities = Embeddings.objects.all()
    for similarity in similarities:
        words.append(similarity.text)
        np_bytes = base64.b64decode(similarity.embedding)
        embeddings.append(pickle.loads(np_bytes))
    pca = PCA(n_components=2)
    result = pca.fit_transform(embeddings)

    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.savefig("plot.png", dpi=1000)


def extract_similarities_for_each_text():
    word_data = list(Words.objects.values_list("text", flat=True))
    save_similarities(word_data)

    print("Word data processing is Done!!!")

    collocation_data = list(Collocations.objects.values_list("text", flat=True))
    save_similarities(collocation_data)

    print("Collocation data processing is Done!!!")

# def konlpy_module(doc, year):
#     measures = nltk.collocations.BigramAssocMeasures()
#     tagged_words = Kkma().sentences(doc)  # only extracts korean nouns
#     pprint("Words: {}".format(tagged_words))
#     print("Words size: {}".format(tagged_words.__len__()))
#
#     finder = nltk.collocations.BigramCollocationFinder.from_words(tagged_words)
#     finder.apply_freq_filter(3)
#     pprint(finder.nbest(measures.pmi, 50))  # top 5 n-grams with highest PMI
#
#     scored = finder.score_ngrams(measures.raw_freq)
#     sorted_list = sorted(scored, key=lambda item: item[1])
#     print("Sorted list: {}".format(sorted_list))
#
#     bigrams = finder.nbest(measures.pmi, 50)
#
#     for bigram in bigrams:
#         new_bigram, created = Trigrams.objects.get_or_create(text=" ".join(map(str, bigram)), year=year)
#         if created:
#             new_bigram.count = 1
#             new_bigram.save()
#         else:
#             new_bigram.count += 1
#             new_bigram.save()


# def word2vec_function(docs_ko):
#
#
#     # t = Okt()
#     # pos = lambda d: ['/'.join(p) for p in t.pos(d)]
#     # tokens = nltk.wordpunct_tokenize(docs_ko)
#     # texts_ko = [pos(doc) for doc in tokens]
#
#     with open('file.txt', 'w+', encoding='utf-8') as f:
#         f.write(docs_ko)
#
#     wv_model_ko = word2vec.Word2Vec(corpus_file='file.txt', window=4)
#     wv_model_ko.init_sims(replace=True)
#     wv_model_ko.save('ko_word2vec.model')
#
#     a = wv_model_ko.wv.most_similar('인공', topn=100)
#     print(a)
