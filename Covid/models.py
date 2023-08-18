from django.db import models

class Photo(models.Model):
    #name = models.CharField(max_length = 25)
    image = models.ImageField(upload_to='photos/')

    # change the settings 

    def __str__(self):
        return self.name
