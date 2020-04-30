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
from mysite.models import Words, Bigrams, Trigrams, WordsSimilarity


class Home(TemplateView):
    template_name = "home.html"


def db_view(request):
    if request.method == 'GET':
        return render(request, 'db_view.html')
    if request.method == 'POST':
        if not request.POST:
            return render(request, 'db_view.html')

        year = request.POST['year']

        data = Words.objects.all().filter(year=year).order_by('-count').values('text', 'count', 'year')
        if data.__len__() == 0:
            return render(request, 'no_data.html')

        context = {}
        context['words'] = data
        context['result'] = 1
        return render(request, 'db_view.html', context)


def visualize(request):
    if request.method == 'GET':
        kmeans_clustering()
        return render(request, 'visualization.html')
    if request.method == 'POST':
        if not request.POST:
            return render(request, 'visualization.html')

        year = request.POST['year']
        wc_dir = 'media/tmp_wordcloud'

        context = {}

        # region Prepare data for Chart
        year_start = int(request.POST['year_start'])
        year_end = int(request.POST['year_end'])
        top5_words_for_sel_year = Words.objects.filter(year=year).order_by('-count')[:5]
        top5_collocations_for_sel_year = Bigrams.objects.filter(year=year).order_by('-count')[:5]

        if not top5_words_for_sel_year.__len__() == 0:
            words = []
            context['word_chart_data'] = []
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
                        txt = w[counter_words - 1].text
                        yr = counter_year
                        cnt = 0

                    context['word_chart_data'].append({"text": txt, "year": yr, "count": cnt})
                    counter_year += 1

            context['is_word_chart_data_exists'] = 1

        if not top5_collocations_for_sel_year.__len__() == 0:
            collocations = []
            context['collocation_chart_data'] = []
            for collocation in top5_collocations_for_sel_year:
                collocations.append(Bigrams.objects.filter(text=collocation.text, year__range=[year_start, year_end]).order_by('year'))

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

                    context['collocation_chart_data'].append({"text": txt, "year": yr, "count": cnt})
                    counter_year += 1

            context['is_collocation_chart_data_exists'] = 1
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

        context['result'] = 1
        return render(request, 'visualization.html', context)


def upload(request):
    context = {}
    if request.method == 'POST':
        if not request.FILES:
            context['result'] = 0
            return render(request, 'upload.html', context)

        fs = FileSystemStorage()
        uploaded_file = request.FILES['document']
        files = os.listdir('media')

        filename = uploaded_file.name

        # remove space in filename if exists
        if " " in filename:
            filename = uploaded_file.name.replace(" ", "")

        if filename in files:
            context['result'] = -1
            return render(request, 'upload.html', context)

        name = fs.save(filename, uploaded_file)

        year = int(request.POST['year'])
        process_result = process_file(name, year)
        context['created_words'] = process_result['created']
        context['updated_words'] = process_result['updated']
        context['result'] = 1
    return render(request, 'upload.html', context)


def process_file(filename, year):
    # reg_exp_date = re.compile(r'([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))')
    # year = 0
    # if reg_exp_date.search(filename):
    #     year = int(reg_exp_date.search(filename).group().split('.')[0])

    parsed = parser.from_file('./media/' + filename)
    print("************ NLTK function ************")
    nltk_function(pre_process(parsed["content"]), year)
    print("************ KONLPY function ************")
    save_similarities(pre_process(parsed["content"]))
    # konlpy_module(parsed["content"], year)
    # word2vec_function(parsed["content"].replace('\n', ' '))
    words_array = re.findall('\\w+', parsed["content"].replace('\n', ''))

    return save_words(words_array, year)


def save_words(words_array, year):
    result = {'created': 0, 'updated': 0}
    for word_item in words_array:
        new_word, created = Words.objects.get_or_create(text=word_item, year=year)
        if created:
            result['created'] += 1
            new_word.count = 1
            new_word.save()
        else:
            result['updated'] += 1
            new_word.count += 1
            new_word.save()
    return result


def kmeans_clustering():
    num_of_clusters = 20
    words = []
    embeddings = []
    similarities = WordsSimilarity.objects.all()
    for similarity in similarities:
        words.append(similarity.text)
        np_bytes = base64.b64decode(similarity.embedding)
        embeddings.append(pickle.loads(np_bytes))

    clustering = KMeans(n_clusters=num_of_clusters)
    print(embeddings.__len__())
    clustering.fit(embeddings)
    for i in range(0, num_of_clusters):
        n_cluster_data = ClusterIndicesNumpy(clustNum=i, labels_array=clustering.labels_)
        print("***** Cluster {} *******".format(i))
        for n in n_cluster_data:
            print("{}".format(words[n]))


def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


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


def pre_process(corpus):
    corpus = corpus.lower()
    stopset = []
    with open("stopwords_ko.txt", 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            stopset.append(line[:-1])

    corpus = " ".join([i for i in nltk.regexp_tokenize(corpus, '\\w+') if i not in stopset])
    return corpus


def save_similarities(corpus):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    tokens = nltk.word_tokenize(corpus)
    for token in tokens:
        input_ids = torch.tensor(tokenizer.encode(token)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
        np_arr = np.array(last_hidden_states[0].detach().numpy())
        np_arr = np.mean(np_arr, axis=0)
        np_bytes = pickle.dumps(np_arr)
        np_base64 = base64.b64encode(np_bytes)
        new_item, created = WordsSimilarity.objects.get_or_create(text=token, embedding=np_base64)
        if created:
            print("Created")
            np_bytes = base64.b64decode(new_item.embedding)
            np_array = pickle.loads(np_bytes)
            print(np_array)

    # print("Tokens string: {}".format(tagged_words))
    # print("Tokens int: {}".format(ids))
    # print("Converted Tokens string: {}".format(converted))
    # print("Tokens string length: {}".format(tagged_words.__len__()))
    # print("Tokens int length: {}".format(ids.__len__()))


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
    similarities = WordsSimilarity.objects.all()
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


def nltk_function(doc_ko, year):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    tokens = nltk.word_tokenize(doc_ko)
    print("Tokenize: {}".format(tokens))

    print("Words size: {}".format(tokens.__len__()))

    finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)
    pprint(finder.nbest(bigram_measures.pmi, 50))
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    sorted_list = sorted(scored, key=lambda item: item[1])
    print("Sorted list: {}".format(sorted_list))

    bigrams = finder.nbest(bigram_measures.pmi, 50)

    for bigram in bigrams:
        new_bigram, created = Bigrams.objects.get_or_create(text=" ".join(map(str, bigram)), year=year)
        if created:
            new_bigram.count = 1
            new_bigram.save()
        else:
            new_bigram.count += 1
            new_bigram.save()

    # print(scored)
    # scores = sorted(score for bigram, score in scored)
    # bigrams = sorted(bigram for bigram, score in scored)
    #
    # print(scores)
    # print(bigrams)

    # t = Okt()
    # tokens_ko = t.morphs(doc_ko)
    # ko = nltk.Text(tokens_ko, name=u'유니코드')  # For Python 2, input `name` as u'유니코드'
    # print(ko.collocation_list())

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
