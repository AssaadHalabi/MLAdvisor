from django.shortcuts import render



def image_demo(request):
    return render(request, 'demo/image.html')



def text_demo(request):
    return render(request, 'demo/text.html')
