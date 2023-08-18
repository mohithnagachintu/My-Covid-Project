from django.forms import ModelForm
from .models import Photo

class PhotoSubmitForm(ModelForm):
    class Meta:
        model = Photo
        fields = ['image']