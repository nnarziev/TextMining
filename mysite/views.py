import os

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.generic import TemplateView
import html2text
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np

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
        uploaded_file = request.FILES['document']
        files = os.listdir('media')

        filename = uploaded_file.name
        # remove space in filename if exists
        if " " in filename:
            filename = uploaded_file.name.replace(" ", "")

        fs = FileSystemStorage()

        if filename in files:
            context['result'] = -1
            return render(request, 'upload.html', context)

        name = fs.save(filename, uploaded_file)
        print(name)

        html_name = os.path.splitext(name)[0] + ".html"
        os.system("hwp5html media\{0} --output=media\{1} --html".format(name, html_name))
        context['url'] = fs.url(name)
        file_path = os.path.join('media', html_name)
        process_result = process_file(file_path)
        context['created_words'] = process_result['created']
        context['updated_words'] = process_result['updated']
        context['result'] = 1
    return render(request, 'upload.html', context)


def process_file(file):
    # Ignore converting links from HTML
    result = {'created': 0,
              'updated': 0}
    with open(file, 'r', encoding='UTF-8', newline='') as f:
        data = f.read().replace('\n', '')
        text = html2text.html2text(data)
        words_array = re.findall('\\w+', text)
        for word_item in words_array:
            new_word, created = Words.objects.get_or_create(word=word_item)
            if created:
                result['created'] += 1
                new_word.save()
            else:
                result['updated'] += 1
                new_word.counter += 1
                new_word.save()

    os.remove(file)
    return result


def db_view(request):
    context = {}
    context['words'] = Words.objects.all().order_by('-counter').values('word', 'counter')
    context['result'] = 1
    return render(request, 'db_view.html', context)


def visualize(request):
    wc_dir = 'media/tmp_wordcloud'

    word_cloud_count = 100
    chart_data_count = 5

    word_freq = {}

    data = Words.objects.all().order_by('-counter').values('word', 'counter')

    for index, item in enumerate(data):
        if index == word_cloud_count:
            break
        word_freq[item['word']] = item['counter']

    mask = np.array(Image.open(os.path.join(wc_dir, "cloud.png")))
    stopwords = set(STOPWORDS)
    wc = WordCloud(mask=mask, font_path='C:\\Users\\NSL\\Downloads\\Nanum_Gothic\\NanumGothic-Regular.ttf', background_color="white",
                   max_words=word_cloud_count,
                   stopwords=stopwords)

    wc.generate_from_frequencies(word_freq)
    wc.to_file(os.path.join(wc_dir, 'wc.png'))
    context = {}
    context['chart_data'] = data[:chart_data_count]
    context['result'] = 1
    return render(request, 'visualization.html', context)
