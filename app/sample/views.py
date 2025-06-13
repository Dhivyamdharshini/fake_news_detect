from django.shortcuts import render
import joblib
import os

vectorizer = joblib.load(os.path.join('ml_model', 'vectorizer.pkl'))
model = joblib.load(os.path.join('ml_model', 'lr_model.pkl'))

def classify_view(request):
    result = None
    news_input = ""

    if request.method == 'POST':
        news_input = request.POST.get('newsText', '').strip()

        if news_input:
            try:
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)[0]

                if prediction == 1:
                    result = "The News is Real!"
                else:
                    result = "The News is Fake!"
            except Exception as e:
                result = f"Error during prediction: {str(e)}"
        else:
            result = "Please enter some text to analyze."

    return render(request, 'classify.html', {'prediction': result, 'news_input': news_input})
