import os

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.generic import TemplateView
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
from tika import parser
from konlpy.tag import Okt
import nltk

# Create your views here.
from mysite.models import Words


class Home(TemplateView):
    template_name = "home.html"


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

        process_result = process_file(name)
        context['created_words'] = process_result['created']
        context['updated_words'] = process_result['updated']
        context['result'] = 1
    return render(request, 'upload.html', context)


def process_file(filename):
    # Ignore converting links from HTML
    reg_exp_date = re.compile(r'([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))')
    year = 0
    if reg_exp_date.search(filename):
        year = int(reg_exp_date.search(filename).group().split('.')[0])

    parsed = parser.from_file('./media/' + filename)
    # nltk_function(parsed["content"])
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


def nltk_function(doc_ko):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    tokens = nltk.regexp_tokenize(doc_ko, '\\w+')
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(4)
    print(finder.nbest(bigram_measures.pmi, 30))
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    sorted_list = sorted(scored, key=lambda item: item[1])
    print(sorted_list)
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


def word2vec_function(docs_ko):
    from konlpy.tag import Okt
    from gensim.models import word2vec

    # t = Okt()
    # pos = lambda d: ['/'.join(p) for p in t.pos(d)]
    # tokens = nltk.wordpunct_tokenize(docs_ko)
    # texts_ko = [pos(doc) for doc in tokens]

    with open('file.txt', 'w+', encoding='utf-8') as f:
        f.write(docs_ko)

    wv_model_ko = word2vec.Word2Vec(corpus_file='file.txt', window=4)
    wv_model_ko.init_sims(replace=True)
    wv_model_ko.save('ko_word2vec.model')

    a = wv_model_ko.wv.most_similar('인공', topn=100)
    print(a)


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
        return render(request, 'visualization.html')
    if request.method == 'POST':
        if not request.POST:
            return render(request, 'visualization.html')

        year = request.POST['year']
        wc_dir = 'media/tmp_wordcloud'

        word_cloud_count = 100
        chart_data_count = 5
        chart_years_count = 12

        word_freq = {}

        data_years = Words.objects.filter(year__range=[int(year) - chart_years_count, int(year)]).values_list('year', flat=True).distinct()
        data_years = sorted(data_years)
        data_years_count = data_years.__len__()

        chart_data = Words.objects.all().filter(year__range=[int(year) - chart_years_count, int(year)]).order_by('-count', 'text', 'year').values('text', 'count', 'year')[:chart_data_count * data_years_count]
        wc_data = Words.objects.all().filter(year=year).order_by('-count').values('text', 'count', 'year')[:word_cloud_count]

        if chart_data.__len__() == 0:
            return render(request, 'no_data.html')

        for item in wc_data:
            word_freq[item['text']] = item['count']

        mask = np.array(Image.open(os.path.join(wc_dir, "cloud.png")))
        stopwords = set(STOPWORDS)
        wc = WordCloud(mask=mask, font_path='C:\\Users\\NSL\\Downloads\\Nanum_Gothic\\NanumGothic-Regular.ttf', background_color="white",
                       max_words=word_cloud_count,
                       stopwords=stopwords)

        context = {}
        if not word_freq.__len__() == 0:
            wc.generate_from_frequencies(word_freq)
            wc.to_file(os.path.join(wc_dir, 'wc.png'))
            context['wc_data'] = 1
        context['chart_data'] = chart_data[:]
        context['result'] = 1
        return render(request, 'visualization.html', context)
