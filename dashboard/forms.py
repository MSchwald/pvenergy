from django import forms

SYSTEM_IDS = [
    ("sys1", "System 1"),
    ("sys2", "System 2"),
    ("sys3", "System 3"),
]

MODEL_CHOICES = [
    ("model1", "Modell 1"),
    ("model2", "Modell 2"),
    ("model3", "Modell 3"),
]

class ForecastForm(forms.Form):
    system_id = forms.ChoiceField(label="System", choices=SYSTEM_IDS)
    model_name = forms.ChoiceField(label="Modell", choices=MODEL_CHOICES)