from django.urls import path
from django.shortcuts import render
from . import views

def view_wrapper(view_func, template_name, active_page):
    def wrapped(request, *args, **kwargs):
        context = view_func(request, *args, **kwargs)
        if context is None:
            context = {}
        return render(request, template_name, {
            **context,
            "active_page": active_page,
        })
    return wrapped

def make_menu_urls(menu: list[tuple[str, str]]) -> list:
    """Takes a list of (label, view) pairs, returns a list of menu urls for url.py"""
    return [
        path(
            label.lower().replace(" ", "_") + "/" if i!= 0 else "",
            view_wrapper(getattr(views, view), f"dashboard/{view}.html", view),
            name = view
        ) for i, (label, view) in enumerate(menu)
    ]