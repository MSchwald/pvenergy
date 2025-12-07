from .urls import MENU

def menu_context(request):
    return {
        "MENU": MENU
    }