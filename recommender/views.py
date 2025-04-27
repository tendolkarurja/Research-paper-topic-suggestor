from django.shortcuts import render

# Create your views here.
def show_recommendations(request):
    if request.method == "POST":
        choice = request.POST.get('ch')
        return render(request, 'recommender/recommend.html', {'selected_choice' : choice})
    